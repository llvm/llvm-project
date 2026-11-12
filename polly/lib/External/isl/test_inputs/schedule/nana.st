domain: "{ sync[i0, i1] : 0 <= i0 <= 64 and 0 <= i1 <= 32; stB[i0, i1] : 0 <= i0 <= 64 and 0 <= i1 <= 64; stC[i0, i1] : 0 <= i0 <= 64 and 0 <= i1 <= 64 }"
child:
  schedule: "[{ sync[i0, i1] -> [(i0)]; stB[i0, i1] -> [(i0)]; stC[i0, i1] -> [(i0)] }, { sync[i0, i1] -> [(64 + i1)]; stB[i0, i1] -> [(i1)]; stC[i0, i1] -> [(i1)] }]"
  permutable: 1
  coincident: [ 1, 0 ]
  child:
    sequence:
    - filter: "{ stB[i0, i1]; stC[i0, i1] }"
      child:
        set:
        - filter: "{ stC[i0, i1] }"
        - filter: "{ stB[i0, i1] }"
    - filter: "{ sync[i0, i1] }"
