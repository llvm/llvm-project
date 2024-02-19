# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+v,+xsfvfnrclipxfqf %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v,+xsfvfnrclipxfqf %s \
# RUN:        | llvm-objdump -d --mattr=+v,+xsfvfnrclipxfqf - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v,+xsfvfnrclipxfqf %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

sf.vfnrclip.xu.f.qf v4, v8, fa2
# CHECK-INST: sf.vfnrclip.xu.f.qf v4, v8, fa2
# CHECK-ENCODING: [0x5b,0x52,0x86,0x8a]
# CHECK-ERROR: instruction requires the following: 'XSfvfnrclipxfqf' (SiFive FP32-to-int8 Ranged Clip Instructions)
# CHECK-UNKNOWN: 5b 52 86 8a <unknown>

sf.vfnrclip.xu.f.qf v4, v8, fa2, v0.t
# CHECK-INST: sf.vfnrclip.xu.f.qf v4, v8, fa2
# CHECK-ENCODING: [0x5b,0x52,0x86,0x88]
# CHECK-ERROR: instruction requires the following: 'XSfvfnrclipxfqf' (SiFive FP32-to-int8 Ranged Clip Instructions)
# CHECK-UNKNOWN: 5b 52 86 88 <unknown>

sf.vfnrclip.x.f.qf v4, v8, fa2
# CHECK-INST: sf.vfnrclip.x.f.qf v4, v8, fa2
# CHECK-ENCODING: [0x5b,0x52,0x86,0x8e]
# CHECK-ERROR: instruction requires the following: 'XSfvfnrclipxfqf' (SiFive FP32-to-int8 Ranged Clip Instructions)
# CHECK-UNKNOWN: 5b 52 86 8e <unknown>

sf.vfnrclip.x.f.qf v4, v8, fa2, v0.t
# CHECK-INST: sf.vfnrclip.x.f.qf v4, v8, fa2
# CHECK-ENCODING: [0x5b,0x52,0x86,0x8c]
# CHECK-ERROR: instruction requires the following: 'XSfvfnrclipxfqf' (SiFive FP32-to-int8 Ranged Clip Instructions)
# CHECK-UNKNOWN: 5b 52 86 8c <unknown>
