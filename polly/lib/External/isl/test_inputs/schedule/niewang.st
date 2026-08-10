domain: "{ S_3[i0, i1, i2, i3, i4, i5, i6] : 0 <= i0 <= 223 and 0 <= i1 <= 223 and 0 <= i2 <= 15 and 0 <= i3 <= 3 and 0 <= i4 <= 6 and 0 <= i5 <= 6 and 0 <= i6 <= 15; group0[i0, i1, i2, i3] : 0 <= i0 <= 3 and 0 <= i1 <= 223 and 0 <= i2 <= 223 and 0 <= i3 <= 15 }"
child:
  sequence:
  - filter: "{ group0[i0, i1, i2, i3] }"
    child:
      schedule: "[{ group0[i0, i1, i2, i3] -> [(i0)] }, { group0[i0, i1, i2, i3] -> [(i1)] }, { group0[i0, i1, i2, i3] -> [(i2)] }, { group0[i0, i1, i2, i3] -> [(i3)] }]"
      permutable: 1
      coincident: [ 1, 1, 1, 1 ]
  - filter: "{ S_3[i0, i1, i2, i3, i4, i5, i6] }"
    child:
      schedule: "[{ S_3[i0, i1, i2, i3, i4, i5, i6] -> [(i0)] }, { S_3[i0, i1, i2, i3, i4, i5, i6] -> [(i1)] }, { S_3[i0, i1, i2, i3, i4, i5, i6] -> [(i2)] }, { S_3[i0, i1, i2, i3, i4, i5, i6] -> [(i3)] }]"
      permutable: 1
      coincident: [ 1, 1, 1, 1 ]
      child:
        schedule: "[{ S_3[i0, i1, i2, i3, i4, i5, i6] -> [(i4)] }]"
        permutable: 1
        coincident: [ 1 ]
        child:
          schedule: "[{ S_3[i0, i1, i2, i3, i4, i5, i6] -> [(i5)] }]"
          permutable: 1
          coincident: [ 1 ]
          child:
            schedule: "[{ S_3[i0, i1, i2, i3, i4, i5, i6] -> [(i6)] }]"
            permutable: 1
            coincident: [ 1 ]
