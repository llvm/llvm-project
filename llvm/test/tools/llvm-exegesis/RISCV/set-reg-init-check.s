# RUN: llvm-exegesis -mode=latency -mtriple=riscv32-unknown-linux-gnu --mcpu=generic --dump-object-to-disk=%d --benchmark-phase=assemble-measured-code --opcode-name=FADD_D -mattr="+d" 2>&1
# RUN: llvm-objdump -M numeric -d %d > %t.s
# RUN: FileCheck %s --check-prefix=FCVT_D_W_ASM < %t.s

FCVT_D_W_ASM:       <foo>:
FCVT_D_W_ASM:       li       x30, 0x0
FCVT_D_W_ASM-NEXT:  fcvt.d.w f{{[0-9]|[12][0-9]|3[01]}}, x30
