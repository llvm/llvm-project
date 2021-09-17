# RUN: llvm-exegesis -mode=uops -opcode-name=CMOV16rm  -benchmarks-file=%t.CMOV16rm-uops.yaml
# RUN: FileCheck -check-prefixes=CHECK-YAML -input-file=%t.CMOV16rm-uops.yaml %s
# RUN: llvm-exegesis -mcpu=bdver2 -mode=analysis -benchmarks-file=%t.CMOV16rm-uops.yaml -analysis-clusters-output-file=- -analysis-clustering-epsilon=0.1 -analysis-inconsistency-epsilon=0.1 -analysis-numpoints=1 -analysis-clustering=naive | FileCheck -check-prefixes=CHECK-CLUSTERS %s

# https://bugs.llvm.org/show_bug.cgi?id=41448
# 1. Verify that we correctly serialize RegNo 0 as %noreg, not as an empty string!
# 2. Verify that deserialization works. Since CMOV16rm has a variant sched class, just printing clusters is sufficient

CHECK-YAML:      ---
CHECK-YAML-NEXT: mode:            uops
CHECK-YAML-NEXT: key:
CHECK-YAML-NEXT:   instructions:
CHECK-YAML-NEXT:     - 'CMOV16rm {{[A-Z0-9]+}} {{[A-Z0-9]+}} {{[A-Z0-9]+}} i_0x1 %noreg i_0x0 %noreg i_0x{{[0-9a-f]}}'
CHECK-YAML-LAST: ...

# CHECK-CLUSTERS: {{^}}cluster_id,opcode_name,config,sched_class,
# CHECK-CLUSTERS-NEXT: {{^}}0,
