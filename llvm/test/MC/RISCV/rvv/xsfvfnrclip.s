# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+v,+xsfvfnrclipxfqf %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --verify -triple=riscv64 -show-encoding %s
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v,+xsfvfnrclipxfqf %s \
# RUN:        | llvm-objdump -d --mattr=+v,+xsfvfnrclipxfqf - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST

sf.vfnrclip.xu.f.qf v4, v8, fa2 # expected-error {{instruction requires the following: 'XSfvfnrclipxfqf' (SiFive FP32-to-int8 Ranged Clip Instructions)}}
# CHECK-INST: sf.vfnrclip.xu.f.qf v4, v8, fa2
# CHECK-ENCODING: [0x5b,0x52,0x86,0x8a]

sf.vfnrclip.xu.f.qf v4, v8, fa2, v0.t # expected-error {{instruction requires the following: 'XSfvfnrclipxfqf' (SiFive FP32-to-int8 Ranged Clip Instructions)}}
# CHECK-INST: sf.vfnrclip.xu.f.qf v4, v8, fa2
# CHECK-ENCODING: [0x5b,0x52,0x86,0x88]

sf.vfnrclip.x.f.qf v4, v8, fa2 # expected-error {{instruction requires the following: 'XSfvfnrclipxfqf' (SiFive FP32-to-int8 Ranged Clip Instructions)}}
# CHECK-INST: sf.vfnrclip.x.f.qf v4, v8, fa2
# CHECK-ENCODING: [0x5b,0x52,0x86,0x8e]

sf.vfnrclip.x.f.qf v4, v8, fa2, v0.t # expected-error {{instruction requires the following: 'XSfvfnrclipxfqf' (SiFive FP32-to-int8 Ranged Clip Instructions)}}
# CHECK-INST: sf.vfnrclip.x.f.qf v4, v8, fa2
# CHECK-ENCODING: [0x5b,0x52,0x86,0x8c]
