# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+experimental-zvfbfa %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+experimental-zvfbfa %s \
# RUN:        | llvm-objdump -d --mattr=+experimental-zvfbfa --no-print-imm-hex - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+experimental-zvfbfa %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

vsetvli a2, a0, e8alt, m1, ta, ma
# CHECK-INST: vsetvli a2, a0, e8alt, m1, ta, ma
# CHECK-ENCODING: [0x57,0x76,0x05,0x1c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 1c057657 <unknown>

vsetvli a2, a0, e16alt, m1, ta, ma
# CHECK-INST: vsetvli a2, a0, e16alt, m1, ta, ma
# CHECK-ENCODING: [0x57,0x76,0x85,0x1c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 1c857657 <unknown>

vsetivli a2, 0, e8alt, m1, ta, ma
# CHECK-INST: vsetivli a2, 0, e8alt, m1, ta, ma
# CHECK-ENCODING: [0x57,0x76,0x00,0xdc]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: dc007657 <unknown>

vsetivli a2, 0, e16alt, m1, ta, ma
# CHECK-INST: vsetivli a2, 0, e16alt, m1, ta, ma
# CHECK-ENCODING: [0x57,0x76,0x80,0xdc]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: dc807657 <unknown>
