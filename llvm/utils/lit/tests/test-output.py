# RUN: %{lit} -j 1 -v %{inputs}/test-data --output %t.results.out > %t.out
# RUN: FileCheck < %t.results.out %s

# CHECK: {
# CHECK: "__version__"
# CHECK: "elapsed"
# CHECK-NEXT: "tests": [
# CHECK-NEXT:   {
# CHECK-NEXT:     "code": "PASS",
# CHECK-NEXT:     "elapsed": {{[0-9.]+}},
# CHECK-NEXT:     "metrics": {
# CHECK-NEXT:       "value0": 1,
# CHECK-NEXT:       "value1": 2.3456
# CHECK-NEXT:     }
# CHECK:     "name": "test-data :: bad&name.ini",
# CHECK:     "output": "& < > \""

# CHECK: ]
# CHECK-NEXT: }
