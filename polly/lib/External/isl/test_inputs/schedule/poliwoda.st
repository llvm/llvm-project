domain: "[_PB_M, _PB_N] -> { S_0[Id1, Id2, Id3] : _PB_M >= 2 and Id1 >= 0 and 4Id1 < _PB_N and 2Id2 >= -_PB_M and 4Id2 <= -_PB_M and Id3 <= 0 and 4Id3 >= -3 + _PB_M + 4Id2 and 4Id3 >= -1 - _PB_M; S_1[Id1, Id2, Id3] : _PB_M >= 2 and Id1 >= 0 and 4Id1 < _PB_N and -_PB_M <= 2Id2 <= 1 - _PB_M and Id3 <= 0 and 4Id3 >= -1 - _PB_M }"
child:
  schedule: "[_PB_M, _PB_N] -> [{ S_0[Id1, Id2, Id3] -> [(Id1)]; S_1[Id1, Id2, Id3] -> [(Id1)] }, { S_0[Id1, Id2, Id3] -> [(-Id2)]; S_1[Id1, Id2, Id3] -> [(_PB_M)] }, { S_0[Id1, Id2, Id3] -> [(-Id3)]; S_1[Id1, Id2, Id3] -> [(-Id3)] }]"
  permutable: 1
  coincident: [ 1, 1, 1 ]
