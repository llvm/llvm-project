# XAndesVSIntH - Andes Vector Small INT Handling Extension
# RUN: not llvm-mc -triple=riscv32 -mattr=+xandesvsinth < %s 2>&1 | FileCheck %s
# RUN: not llvm-mc -triple=riscv64 -mattr=+xandesvsinth < %s 2>&1 | FileCheck %s

# CHECK: :[[@LINE+1]]:1: error: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32f' (Vector Extensions for Embedded Processors){{$}}
nds.vfwcvt.f.n.v v8, v10

# CHECK: :[[@LINE+1]]:1: error: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32f' (Vector Extensions for Embedded Processors){{$}}
nds.vfwcvt.f.nu.v v8, v10

# CHECK: :[[@LINE+1]]:1: error: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32f' (Vector Extensions for Embedded Processors){{$}}
nds.vfwcvt.f.b.v v8, v10

# CHECK: :[[@LINE+1]]:1: error: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32f' (Vector Extensions for Embedded Processors){{$}}
nds.vfwcvt.f.bu.v v8, v10
