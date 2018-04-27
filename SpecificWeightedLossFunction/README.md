# Weighted Loss Function For Specific Prediction


|    | c1 | c2 | c3 | c4 |
| -- | -- | -- | -- | -- | 
| c1  |  0.3  | 0.4  | 0.2 | 0.1 |
| c2  |  0.3  | 0.5  | 0.1 | 0.1 |
| c3  |  0.1  | 0.1  | 0.7 | 0.1 |
| c4  |  0.2  | 0.1  | 0.4 | 0.3 |

- For above confusion matrix, all predictions can be evaluated with different weights. For example , c4-c3( 0.4 ) misprediction ratio can
have more penalty than c4-c2( 0.1 ) misprediction's.  For n classes, you have n x n weight matrix , and you initialize it as you wish.

- This script calculate confusion matrix for validation data. Then , set weight matrix element according to this validation matrix after
every  epoch.


Reference : https://github.com/keras-team/keras/issues/2115
