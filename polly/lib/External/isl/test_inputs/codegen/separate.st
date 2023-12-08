domain: "{ a[i] : 0 <= i < 10; b[i] : 0 <= i < 10 }"
child:
  schedule: "[{ a[i] -> [i]; b[i] -> [i+1] }]"
  options: "{ separate[x] }"
  child:
    sequence:
    - filter: "{ a[i] }"
    - filter: "{ b[i] }"
