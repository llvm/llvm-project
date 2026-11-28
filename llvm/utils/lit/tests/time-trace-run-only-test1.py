# RUN: rm -fr %t.json
# RUN: %{lit} --filter 'test1' --time-trace-output %t.json %{inputs}/time-trace
# RUN: FileCheck < %t.json %s

# CHECK: {
# CHECK-NEXT:   "traceEvents": [
# CHECK-NEXT: {
# CHECK-NOT: "name": "time-trace :: test2.txt",
# CHECK-DAG: "name": "time-trace :: test1.txt",
