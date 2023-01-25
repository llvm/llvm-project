# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=latency --benchmark-phase=assemble-measured-code       -opcode-name=LEA64r -repetition-mode=duplicate | FileCheck %s --check-prefixes=CHECK,CHECK-CODEGEN
# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=latency --benchmark-phase=prepare-snippet              -opcode-name=LEA64r -repetition-mode=duplicate | FileCheck %s --check-prefixes=CHECK,CHECK-NO-CODEGEN
# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=latency --benchmark-phase=prepare-and-assemble-snippet -opcode-name=LEA64r -repetition-mode=duplicate | FileCheck %s --check-prefixes=CHECK,CHECK-CODEGEN

CHECK:      ---
CHECK-NEXT: mode: latency
CHECK-NEXT: key:
CHECK-NEXT:   instructions:
CHECK-NEXT:     LEA64r
CHECK-CODEGEN: assembled_snippet: {{[A-Z0-9]+}}{{$}}
CHECK-NO-CODEGEN: assembled_snippet: ''{{$}}
