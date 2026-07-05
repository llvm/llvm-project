# Check that component detection is not confused by values
# of the schedule dimension that do not correspond to any statement instances.
domain: "[K] -> { S[]; T[i] : 0 <= i < K }"
child:
  context: "[K] -> { [] : K > 0 }"
  child:
    schedule: "[K] -> [{ S[] -> [(0)]; T[i] -> [(32*floor((i)/32))] }]"
    child:
      sequence:
      - filter: "[K] -> { S[] }"
      - filter: "[K] -> { T[i] }"
