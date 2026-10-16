# Check that the generated code does not involve any loop
# since S has only a single instance.
domain: "[s = 0:23] -> { S[a = 0:2, b = 0:3, c] : 12c = s - 4a - b }"
child:
  context: "[s] -> { [] : 0 <= s <= 23 }"
  child:
    schedule: "[s] -> S[{ S[a, b, c] -> [a] }, { S[a, b, c] -> [b] }, { S[a, b, c] -> [c] }]"
    options: "{ atomic[*] }"
