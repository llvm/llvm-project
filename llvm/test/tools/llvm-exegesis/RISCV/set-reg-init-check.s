# RUN: llvm-exegesis -mode=latency -mtriple=riscv32-unknown-linux-gnu --mcpu=generic --dump-object-to-disk=%t.o --benchmark-phase=assemble-measured-code --opcode-name=FADD_D -mattr="+d" 2>&1
# RUN: llvm-objdump -M numeric -d %t.o > %t.s
# RUN: FileCheck %s < %t.s

CHECK:       <foo>:
CHECK:       li       x30, 0x0
CHECK-NEXT:  fcvt.d.w f{{[0-9]|[12][0-9]|3[01]}}, x30
