# RUN: llvm-mc -triple=hexagon -mcpu=hexagonv5 --filetype=obj %s -o - | llvm-readelf --file-headers - | FileCheck --check-prefix=CHECK-V5 %s
# RUN: llvm-mc -triple=hexagon -mcpu=hexagonv55 --filetype=obj %s -o - | llvm-readelf --file-headers - | FileCheck --check-prefix=CHECK-V55 %s
# RUN: llvm-mc -triple=hexagon -mcpu=hexagonv60 --filetype=obj %s -o - | llvm-readelf --file-headers - | FileCheck --check-prefix=CHECK-V60 %s
# RUN: llvm-mc -triple=hexagon -mcpu=hexagonv62 --filetype=obj %s -o - | llvm-readelf --file-headers - | FileCheck --check-prefix=CHECK-V62 %s

# CHECK-V5: Flags: 0x4
# CHECK-V55: Flags: 0x5
# CHECK-V60: Flags: 0x60
# CHECK-V62: Flags: 0x62
