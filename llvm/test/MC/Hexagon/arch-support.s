# RUN: llvm-mc -triple=hexagon -mv5 -filetype=obj %s | llvm-readelf -h - | FileCheck --check-prefix=CHECK-V5 %s
# RUN: llvm-mc -triple=hexagon -mv55 -filetype=obj %s | llvm-readelf -h - | FileCheck --check-prefix=CHECK-V55 %s
# RUN: llvm-mc -triple=hexagon -mv60 -filetype=obj %s | llvm-readelf -h - | FileCheck --check-prefix=CHECK-V60 %s
# RUN: llvm-mc -triple=hexagon -mv62 -filetype=obj %s | llvm-readelf -h - | FileCheck --check-prefix=CHECK-V62 %s
# RUN: llvm-mc -triple=hexagon -mv65 -filetype=obj %s | llvm-readelf -h - | FileCheck --check-prefix=CHECK-V65 %s
# RUN: llvm-mc -triple=hexagon -mv67 -filetype=obj %s | llvm-readelf -h - | FileCheck --check-prefix=CHECK-V67 %s
# RUN: llvm-mc -triple=hexagon -mv68 -filetype=obj %s | llvm-readelf -h - | FileCheck --check-prefix=CHECK-V68 %s
# RUN: llvm-mc -triple=hexagon -mv69 -filetype=obj %s | llvm-readelf -h - | FileCheck --check-prefix=CHECK-V69 %s

# RUN: llvm-mc -triple=hexagon -mv5 -filetype=obj %s | llvm-objdump --disassemble - | FileCheck --check-prefix=CHECK-OBJDUMP %s
# RUN: llvm-mc -triple=hexagon -mv55 -filetype=obj %s | llvm-objdump --disassemble - | FileCheck --check-prefix=CHECK-OBJDUMP %s
# RUN: llvm-mc -triple=hexagon -mv60 -filetype=obj %s | llvm-objdump --disassemble - | FileCheck --check-prefix=CHECK-OBJDUMP %s
# RUN: llvm-mc -triple=hexagon -mv62 -filetype=obj %s | llvm-objdump --disassemble - | FileCheck --check-prefix=CHECK-OBJDUMP %s
# RUN: llvm-mc -triple=hexagon -mv65 -filetype=obj %s | llvm-objdump --disassemble - | FileCheck --check-prefix=CHECK-OBJDUMP %s
# RUN: llvm-mc -triple=hexagon -mv67 -filetype=obj %s | llvm-objdump --disassemble - | FileCheck --check-prefix=CHECK-OBJDUMP %s
# RUN: llvm-mc -triple=hexagon -mv68 -filetype=obj %s | llvm-objdump --disassemble - | FileCheck --check-prefix=CHECK-OBJDUMP %s
# RUN: llvm-mc -triple=hexagon -mv69 -filetype=obj %s | llvm-objdump --disassemble - | FileCheck --check-prefix=CHECK-OBJDUMP %s
    .text
r1 = r1

# CHECK-V5: Flags:{{.*}}0x4
# CHECK-V55: Flags:{{.*}}0x5
# CHECK-V60: Flags:{{.*}}0x60
# CHECK-V62: Flags:{{.*}}0x62
# CHECK-V65: Flags:{{.*}}0x65
# CHECK-V67: Flags:{{.*}}0x67
# CHECK-V68: Flags:{{.*}}0x68
# CHECK-V69: Flags:{{.*}}0x69

# CHECK-OBJDUMP: { r1 = r1 }
