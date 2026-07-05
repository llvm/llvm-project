domain: "{ S_6[i, j] : i >= 0 and i < j <= 10; S_4[i, j, k] : i >= 0 and i < j <= 10 and 0 <= k <= 100; S_2[i, j] : i >= 0 and i < j <= 10 }"
child:
  schedule: "[{ S_6[i, j] -> [(0)]; S_4[i, j, k] -> [(i)]; S_2[i, j] -> [(0)] }, { S_6[i, j] -> [(10 + i)]; S_4[i, j, k] -> [(j)]; S_2[i, j] -> [(i)] }, { S_6[i, j] -> [(j)]; S_4[i, j, k] -> [(10 - k)]; S_2[i, j] -> [(j)] }]"
  permutable: 1
  coincident: [ 1, 1, 1 ]
