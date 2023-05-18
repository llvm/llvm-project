# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=latency --benchmark-phase=assemble-measured-code -opcode-name=ADDPSrr -repetition-mode=duplicate -dump-object-to-disk 2>&1 | FileCheck %s --check-prefix=CHECK-ON
# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=latency --benchmark-phase=assemble-measured-code -opcode-name=ADDPSrr -repetition-mode=loop      -dump-object-to-disk 2>&1 | FileCheck %s --check-prefix=CHECK-ON
# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=latency --benchmark-phase=assemble-measured-code -opcode-name=ADDPSrr -repetition-mode=duplicate 2>&1 | FileCheck %s --check-prefix=CHECK-OFF
# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=latency --benchmark-phase=assemble-measured-code -opcode-name=ADDPSrr -repetition-mode=loop  2>&1 | FileCheck %s --check-prefix=CHECK-OFF
# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=latency --benchmark-phase=assemble-measured-code -opcode-name=ADDPSrr -repetition-mode=duplicate 2>&1 | FileCheck %s --check-prefix=CHECK-OFF
# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=latency --benchmark-phase=assemble-measured-code -opcode-name=ADDPSrr -repetition-mode=loop 2>&1 | FileCheck %s --check-prefix=CHECK-OFF

# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=latency --benchmark-phase=prepare-and-assemble-snippet -opcode-name=ADDPSrr -repetition-mode=duplicate 2>&1 | FileCheck %s --check-prefix=CHECK-OFF
# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=latency --benchmark-phase=prepare-and-assemble-snippet -opcode-name=ADDPSrr -repetition-mode=loop 2>&1 | FileCheck %s --check-prefix=CHECK-OFF
# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=latency --benchmark-phase=prepare-and-assemble-snippet -opcode-name=ADDPSrr -repetition-mode=duplicate 2>&1 | FileCheck %s --check-prefix=CHECK-OFF
# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=latency --benchmark-phase=prepare-and-assemble-snippet -opcode-name=ADDPSrr -repetition-mode=loop 2>&1 | FileCheck %s --check-prefix=CHECK-OFF

# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=latency --benchmark-phase=prepare-snippet -opcode-name=ADDPSrr -repetition-mode=duplicate 2>&1 | FileCheck %s --check-prefix=CHECK-OFF
# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=latency --benchmark-phase=prepare-snippet -opcode-name=ADDPSrr -repetition-mode=loop 2>&1 | FileCheck %s --check-prefix=CHECK-OFF
# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=latency --benchmark-phase=prepare-snippet -opcode-name=ADDPSrr -repetition-mode=duplicate 2>&1 | FileCheck %s --check-prefix=CHECK-OFF
# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=latency --benchmark-phase=prepare-snippet -opcode-name=ADDPSrr -repetition-mode=loop 2>&1 | FileCheck %s --check-prefix=CHECK-OFF

# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=latency --benchmark-phase=assemble-measured-code -opcode-name=ADDPSrr -repetition-mode=duplicate -dump-object-to-disk=%t.mydmpfile.o 2>&1 | FileCheck %s --check-prefix=DUMP-FILE-NAME
# RUN: llvm-objdump -d %t.mydmpfile.o | FileCheck %s --check-prefix=ASM
CHECK-ON: Check generated assembly with
CHECK-OFF-NOT: Check generated assembly with
DUMP-FILE-NAME: Check generated assembly with: {{.*}}objdump -d {{.*}}mydmpfile
ASM: addps
ASM: retq
