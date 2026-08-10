# In earlier versions of isl, this test case would cause
# the loop coalescing avoidance to break down because one of the dimensions
# (Id2 in S_1) has a fixed value in terms of the other dimensions.
# Check that this no longer happens.
# The actual schedule that is produced is not that important,
# but it is different depending on whether the whole-component scheduler
# is being used, so pick a particular setting.
# OPTIONS: --schedule-whole-component
domain: [_PB_M, _PB_N] -> { S_0[Id1, Id2, Id3] : _PB_M >= 2 and Id1 >=
0 and 4Id1 < _PB_N and 2Id2 >= -_PB_M and 4Id2 <= -_PB_M and Id3 <=
0 and 4Id3 >= -3 + _PB_M + 4Id2 and 4Id3 >= -1 - _PB_M; S_1[Id1, Id2,
Id3] : _PB_M >= 2 and Id1 >= 0 and 4Id1 < _PB_N and -_PB_M <= 2Id2 <=
1 - _PB_M and Id3 <= 0 and 4Id3 >= -1 - _PB_M }
validity: [_PB_M, _PB_N] -> { S_0[Id1, Id2, Id3] -> S_0[Id1' = Id1, Id2',
Id3'] : Id1 >= 0 and 4Id1 < _PB_N and Id3 <= 0 and 4Id3 >= -3 + _PB_M +
4Id2 and Id2' <= Id2 and 2Id2' >= -_PB_M and Id3' < 0 and Id3' <= Id3
and Id3' < Id2 + Id3 - Id2' and 4Id3' >= -4 + _PB_M + 4Id2 and 4Id3' >=
-3 + _PB_M + 3Id2 + Id2' and -1 - _PB_M <= 4Id3' <= 2 + _PB_M + 4Id2;
S_0[Id1, Id2, Id3] -> S_0[Id1' = Id1, Id2', Id3' = Id3] : Id1 >= 0 and
4Id1 < _PB_N and 4Id2 <= -_PB_M and Id3 <= 0 and 4Id3 >= -3 + _PB_M +
4Id2 and Id2' < Id2 and 2Id2' >= -_PB_M; S_0[Id1, Id2, Id3] -> S_1[Id1'
= Id1, Id2', Id3'] : Id1 >= 0 and 4Id1 < _PB_N and Id3 <= 0 and 4Id3 >=
-3 + _PB_M + 4Id2 and Id2' < Id2 and -_PB_M <= 2Id2' <= 1 - _PB_M and
Id3' < 0 and Id3' <= Id3 and -4 + _PB_M + 4Id2 <= 4Id3' <= 2 + _PB_M +
4Id2; S_0[Id1, Id2, Id3] -> S_1[Id1' = Id1, Id2' = Id2, Id3'] : Id1 >=
0 and 4Id1 < _PB_N and -_PB_M <= 2Id2 <= 1 - _PB_M and Id3 <= 0 and Id3'
< Id3 and -1 - _PB_M <= 4Id3' <= 2 + _PB_M + 4Id2; S_0[Id1, Id2, Id3]
-> S_1[Id1' = Id1, Id2', Id3' = Id3] : Id1 >= 0 and 4Id1 < _PB_N and
4Id2 <= -_PB_M and Id3 <= 0 and 4Id3 >= -3 + _PB_M + 4Id2 and Id2' <
Id2 and -_PB_M <= 2Id2' <= 1 - _PB_M }
proximity: [_PB_M, _PB_N] -> { S_0[Id1, Id2, Id3] -> S_0[Id1' = Id1, Id2',
Id3'] : Id1 >= 0 and 4Id1 < _PB_N and Id3 <= 0 and 4Id3 >= -3 + _PB_M +
4Id2 and Id2' <= Id2 and 2Id2' >= -_PB_M and Id3' < 0 and Id3' <= Id3
and Id3' < Id2 + Id3 - Id2' and 4Id3' >= -4 + _PB_M + 4Id2 and 4Id3' >=
-3 + _PB_M + 3Id2 + Id2' and -1 - _PB_M <= 4Id3' <= 2 + _PB_M + 4Id2;
S_0[Id1, Id2, Id3] -> S_0[Id1' = Id1, Id2', Id3' = Id3] : Id1 >= 0 and
4Id1 < _PB_N and 4Id2 <= -_PB_M and Id3 <= 0 and 4Id3 >= -3 + _PB_M +
4Id2 and Id2' < Id2 and 2Id2' >= -_PB_M; S_0[Id1, Id2, Id3] -> S_1[Id1'
= Id1, Id2', Id3'] : Id1 >= 0 and 4Id1 < _PB_N and Id3 <= 0 and 4Id3 >=
-3 + _PB_M + 4Id2 and Id2' < Id2 and -_PB_M <= 2Id2' <= 1 - _PB_M and
Id3' < 0 and Id3' <= Id3 and -4 + _PB_M + 4Id2 <= 4Id3' <= 2 + _PB_M +
4Id2; S_0[Id1, Id2, Id3] -> S_1[Id1' = Id1, Id2' = Id2, Id3'] : Id1 >=
0 and 4Id1 < _PB_N and -_PB_M <= 2Id2 <= 1 - _PB_M and Id3 <= 0 and Id3'
< Id3 and -1 - _PB_M <= 4Id3' <= 2 + _PB_M + 4Id2; S_0[Id1, Id2, Id3]
-> S_1[Id1' = Id1, Id2', Id3' = Id3] : Id1 >= 0 and 4Id1 < _PB_N and
4Id2 <= -_PB_M and Id3 <= 0 and 4Id3 >= -3 + _PB_M + 4Id2 and Id2' <
Id2 and -_PB_M <= 2Id2' <= 1 - _PB_M }
