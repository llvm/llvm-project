# RUN: llvm-mc -triple=hexagon -mcpu=hexagonv62 -filetype=obj %s | llvm-objdump --no-print-imm-hex --mcpu=hexagonv62 -d - | FileCheck %s --check-prefix=CHECK-V62
# RUN: llvm-mc -triple=hexagon -mcpu=hexagonv65 -filetype=obj %s | llvm-objdump --no-print-imm-hex --mcpu=hexagonv65 -d - | FileCheck %s --check-prefix=CHECK-V65

# CHECK-V62: trap1(#0)
# CHECK-V65: trap1(r0,#0)
trap1(#0)
