# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=latency --benchmark-phase=assemble-measured-code -opcode-name=ADDPSrr -repetition-mode=duplicate -dump-object-to-disk=1 2>&1 | FileCheck %s --check-prefix=CHECK-ON
# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=latency --benchmark-phase=assemble-measured-code -opcode-name=ADDPSrr -repetition-mode=loop      -dump-object-to-disk=1 2>&1 | FileCheck %s --check-prefix=CHECK-ON
# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=latency --benchmark-phase=assemble-measured-code -opcode-name=ADDPSrr -repetition-mode=duplicate -dump-object-to-disk=0 2>&1 | FileCheck %s --check-prefix=CHECK-OFF
# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=latency --benchmark-phase=assemble-measured-code -opcode-name=ADDPSrr -repetition-mode=loop      -dump-object-to-disk=0 2>&1 | FileCheck %s --check-prefix=CHECK-OFF
# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=latency --benchmark-phase=assemble-measured-code -opcode-name=ADDPSrr -repetition-mode=duplicate 2>&1 | FileCheck %s --check-prefix=CHECK-OFF
# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=latency --benchmark-phase=assemble-measured-code -opcode-name=ADDPSrr -repetition-mode=loop 2>&1 | FileCheck %s --check-prefix=CHECK-OFF

# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=latency --benchmark-phase=prepare-and-assemble-snippet -opcode-name=ADDPSrr -repetition-mode=duplicate -dump-object-to-disk=1 2>&1 | FileCheck %s --check-prefix=CHECK-OFF
# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=latency --benchmark-phase=prepare-and-assemble-snippet -opcode-name=ADDPSrr -repetition-mode=loop      -dump-object-to-disk=1 2>&1 | FileCheck %s --check-prefix=CHECK-OFF
# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=latency --benchmark-phase=prepare-and-assemble-snippet -opcode-name=ADDPSrr -repetition-mode=duplicate -dump-object-to-disk=0 2>&1 | FileCheck %s --check-prefix=CHECK-OFF
# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=latency --benchmark-phase=prepare-and-assemble-snippet -opcode-name=ADDPSrr -repetition-mode=loop      -dump-object-to-disk=0 2>&1 | FileCheck %s --check-prefix=CHECK-OFF
# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=latency --benchmark-phase=prepare-and-assemble-snippet -opcode-name=ADDPSrr -repetition-mode=duplicate 2>&1 | FileCheck %s --check-prefix=CHECK-OFF
# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=latency --benchmark-phase=prepare-and-assemble-snippet -opcode-name=ADDPSrr -repetition-mode=loop 2>&1 | FileCheck %s --check-prefix=CHECK-OFF

# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=latency --benchmark-phase=prepare-snippet -opcode-name=ADDPSrr -repetition-mode=duplicate -dump-object-to-disk=1 2>&1 | FileCheck %s --check-prefix=CHECK-OFF
# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=latency --benchmark-phase=prepare-snippet -opcode-name=ADDPSrr -repetition-mode=loop      -dump-object-to-disk=1 2>&1 | FileCheck %s --check-prefix=CHECK-OFF
# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=latency --benchmark-phase=prepare-snippet -opcode-name=ADDPSrr -repetition-mode=duplicate -dump-object-to-disk=0 2>&1 | FileCheck %s --check-prefix=CHECK-OFF
# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=latency --benchmark-phase=prepare-snippet -opcode-name=ADDPSrr -repetition-mode=loop      -dump-object-to-disk=0 2>&1 | FileCheck %s --check-prefix=CHECK-OFF
# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=latency --benchmark-phase=prepare-snippet -opcode-name=ADDPSrr -repetition-mode=duplicate 2>&1 | FileCheck %s --check-prefix=CHECK-OFF
# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=latency --benchmark-phase=prepare-snippet -opcode-name=ADDPSrr -repetition-mode=loop 2>&1 | FileCheck %s --check-prefix=CHECK-OFF

CHECK-ON: Check generated assembly with
CHECK-OFF-NOT: Check generated assembly with
