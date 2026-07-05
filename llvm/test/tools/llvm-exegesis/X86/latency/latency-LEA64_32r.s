# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=latency --benchmark-phase=assemble-measured-code -opcode-name=LEA64_32r -repetition-mode=duplicate -max-configs-per-opcode=2 | FileCheck %s
# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=latency --benchmark-phase=assemble-measured-code -opcode-name=LEA64_32r -repetition-mode=duplicate -max-configs-per-opcode=2 | FileCheck --check-prefix CHECK-COUNTS %s

# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=latency --benchmark-phase=assemble-measured-code -opcode-name=LEA64_32r -repetition-mode=loop -max-configs-per-opcode=2 | FileCheck %s
# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=latency --benchmark-phase=assemble-measured-code -opcode-name=LEA64_32r -repetition-mode=loop -max-configs-per-opcode=2 | FileCheck --check-prefix CHECK-COUNTS %s

## Intentionally run llvm-exegesis twice per output!
# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=latency --benchmark-phase=assemble-measured-code -opcode-name=LEA64_32r -repetition-mode=duplicate -max-configs-per-opcode=2 --benchmarks-file=%t.duplicate.yaml
# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=latency --benchmark-phase=assemble-measured-code -opcode-name=LEA64_32r -repetition-mode=duplicate -max-configs-per-opcode=2 --benchmarks-file=%t.duplicate.yaml
# RUN: FileCheck --input-file %t.duplicate.yaml %s
# RUN: FileCheck --input-file %t.duplicate.yaml --check-prefix CHECK-COUNTS %s
# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=latency --benchmark-phase=assemble-measured-code -opcode-name=LEA64_32r -repetition-mode=loop -max-configs-per-opcode=2 --benchmarks-file=%t.loop.yaml
# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=latency --benchmark-phase=assemble-measured-code -opcode-name=LEA64_32r -repetition-mode=loop -max-configs-per-opcode=2 --benchmarks-file=%t.loop.yaml
# RUN: FileCheck --input-file %t.loop.yaml %s
# RUN: FileCheck --input-file %t.loop.yaml --check-prefix CHECK-COUNTS %s


CHECK:      ---
CHECK-NEXT: mode: latency
CHECK-NEXT: key:
CHECK-NEXT:   instructions:
CHECK-NEXT:     LEA64_32r
CHECK-NEXT: config: '0(%[[REG1:[A-Z0-9]+]], %[[REG1]], 1)'

CHECK:      ---
CHECK-NEXT: mode: latency
CHECK-NEXT: key:
CHECK-NEXT:   instructions:
CHECK-NEXT:     LEA64_32r
CHECK-NEXT: config: '42(%[[REG2:[A-Z0-9]+]], %[[REG2]], 1)'

## Check that we empty our output file once per llvm-exegesis run.
## But not once per benchmark.
CHECK-COUNTS-COUNT-2: mode: latency
