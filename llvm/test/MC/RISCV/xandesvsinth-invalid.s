# XAndesVSIntH - Andes Vector Small INT Handling Extension
# RUN: llvm-mc --verify -triple=riscv32 -mattr=+xandesvsinth < %s
# RUN: llvm-mc --verify -triple=riscv64 -mattr=+xandesvsinth < %s

nds.vfwcvt.f.n.v v8, v10 # expected-error:1 {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32f' (Vector Extensions for Embedded Processors)}}

nds.vfwcvt.f.nu.v v8, v10 # expected-error:1 {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32f' (Vector Extensions for Embedded Processors)}}

nds.vfwcvt.f.b.v v8, v10 # expected-error:1 {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32f' (Vector Extensions for Embedded Processors)}}

nds.vfwcvt.f.bu.v v8, v10 # expected-error:1 {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32f' (Vector Extensions for Embedded Processors)}}
