; RUN: llc < %s -verify-machineinstrs -mtriple=arm64-eabi -aarch64-neon-syntax=apple -aarch64-enable-simd-scalar=true -asm-verbose=false -disable-adv-copy-opt=true | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-NOOPT
; RUN: llc < %s -verify-machineinstrs -mtriple=arm64-eabi -aarch64-neon-syntax=apple -aarch64-enable-simd-scalar=true -asm-verbose=false -disable-adv-copy-opt=false | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-OPT
; RUN: llc < %s -verify-machineinstrs -mtriple=arm64-eabi -aarch64-neon-syntax=generic -aarch64-enable-simd-scalar=true -asm-verbose=false -disable-adv-copy-opt=true | FileCheck %s -check-prefix=GENERIC -check-prefix=GENERIC-NOOPT
; RUN: llc < %s -verify-machineinstrs -mtriple=arm64-eabi -aarch64-neon-syntax=generic -aarch64-enable-simd-scalar=true -asm-verbose=false -disable-adv-copy-opt=false | FileCheck %s -check-prefix=GENERIC -check-prefix=GENERIC-OPT

define <2 x i64> @bar(<2 x i64> %a, <2 x i64> %b) nounwind readnone {
; CHECK-LABEL: bar:
; CHECK: add.2d	v[[REG:[0-9]+]], v0, v1
; CHECK: add	d[[REG3:[0-9]+]], d[[REG]], d1
; CHECK: sub	d[[REG2:[0-9]+]], d[[REG]], d1
; Without advanced copy optimization, we end up with cross register
; banks copies that cannot be coalesced.
; CHECK-NOOPT: fmov [[COPY_REG3:x[0-9]+]], d[[REG3]]
; With advanced copy optimization, we end up with just one copy
; to insert the computed high part into the V register. 
; CHECK-OPT-NOT: fmov
; CHECK: fmov [[COPY_REG2:x[0-9]+]], d[[REG2]]
; CHECK-NOOPT: fmov d0, [[COPY_REG3]]
; CHECK-OPT-NOT: fmov
; CHECK: mov.d v0[1], [[COPY_REG2]]
; CHECK-NEXT: ret
;
; GENERIC-LABEL: bar:
; GENERIC: add	v[[REG:[0-9]+]].2d, v0.2d, v1.2d
; GENERIC: add	d[[REG3:[0-9]+]], d[[REG]], d1
; GENERIC: sub	d[[REG2:[0-9]+]], d[[REG]], d1
; GENERIC-NOOPT: fmov [[COPY_REG3:x[0-9]+]], d[[REG3]]
; GENERIC-OPT-NOT: fmov
; GENERIC: fmov [[COPY_REG2:x[0-9]+]], d[[REG2]]
; GENERIC-NOOPT: fmov d0, [[COPY_REG3]]
; GENERIC-OPT-NOT: fmov
; GENERIC: mov v0.d[1], [[COPY_REG2]]
; GENERIC-NEXT: ret
  %add = add <2 x i64> %a, %b
  %vgetq_lane = extractelement <2 x i64> %add, i32 0
  %vgetq_lane2 = extractelement <2 x i64> %b, i32 0
  %add3 = add i64 %vgetq_lane, %vgetq_lane2
  %sub = sub i64 %vgetq_lane, %vgetq_lane2
  %vecinit = insertelement <2 x i64> undef, i64 %add3, i32 0
  %vecinit8 = insertelement <2 x i64> %vecinit, i64 %sub, i32 1
  ret <2 x i64> %vecinit8
}

define double @subdd_su64(<2 x i64> %a, <2 x i64> %b) nounwind readnone {
; CHECK-LABEL: subdd_su64:
; CHECK: sub d0, d1, d0
; CHECK-NEXT: ret
; GENERIC-LABEL: subdd_su64:
; GENERIC: sub d0, d1, d0
; GENERIC-NEXT: ret
  %vecext = extractelement <2 x i64> %a, i32 0
  %vecext1 = extractelement <2 x i64> %b, i32 0
  %sub.i = sub nsw i64 %vecext1, %vecext
  %retval = bitcast i64 %sub.i to double
  ret double %retval
}

define double @vaddd_su64(<2 x i64> %a, <2 x i64> %b) nounwind readnone {
; CHECK-LABEL: vaddd_su64:
; CHECK: add d0, d1, d0
; CHECK-NEXT: ret
; GENERIC-LABEL: vaddd_su64:
; GENERIC: add d0, d1, d0
; GENERIC-NEXT: ret
  %vecext = extractelement <2 x i64> %a, i32 0
  %vecext1 = extractelement <2 x i64> %b, i32 0
  %add.i = add nsw i64 %vecext1, %vecext
  %retval = bitcast i64 %add.i to double
  ret double %retval
}

; sub MI doesn't access dsub register.
define double @add_sub_su64(<2 x i64> %a, <2 x i64> %b) nounwind readnone {
; CHECK-LABEL: add_sub_su64:
; CHECK: add d0, d1, d0
; CHECK: sub d0, {{d[0-9]+}}, d0
; CHECK-NEXT: ret
; GENERIC-LABEL: add_sub_su64:
; GENERIC: add d0, d1, d0
; GENERIC: sub d0, {{d[0-9]+}}, d0
; GENERIC-NEXT: ret
  %vecext = extractelement <2 x i64> %a, i32 0
  %vecext1 = extractelement <2 x i64> %b, i32 0
  %add.i = add i64 %vecext1, %vecext
  %sub.i = sub i64 0, %add.i
  %retval = bitcast i64 %sub.i to double
  ret double %retval
}
define double @and_su64(<2 x i64> %a, <2 x i64> %b) nounwind readnone {
; CHECK-LABEL: and_su64:
; CHECK: and.8b v0, v1, v0
; CHECK-NEXT: ret
; GENERIC-LABEL: and_su64:
; GENERIC: and v0.8b, v1.8b, v0.8b
; GENERIC-NEXT: ret
  %vecext = extractelement <2 x i64> %a, i32 0
  %vecext1 = extractelement <2 x i64> %b, i32 0
  %or.i = and i64 %vecext1, %vecext
  %retval = bitcast i64 %or.i to double
  ret double %retval
}

define double @orr_su64(<2 x i64> %a, <2 x i64> %b) nounwind readnone {
; CHECK-LABEL: orr_su64:
; CHECK: orr.8b v0, v1, v0
; CHECK-NEXT: ret
; GENERIC-LABEL: orr_su64:
; GENERIC: orr v0.8b, v1.8b, v0.8b
; GENERIC-NEXT: ret
  %vecext = extractelement <2 x i64> %a, i32 0
  %vecext1 = extractelement <2 x i64> %b, i32 0
  %or.i = or i64 %vecext1, %vecext
  %retval = bitcast i64 %or.i to double
  ret double %retval
}

define double @xorr_su64(<2 x i64> %a, <2 x i64> %b) nounwind readnone {
; CHECK-LABEL: xorr_su64:
; CHECK: eor.8b v0, v1, v0
; CHECK-NEXT: ret
; GENERIC-LABEL: xorr_su64:
; GENERIC: eor v0.8b, v1.8b, v0.8b
; GENERIC-NEXT: ret
  %vecext = extractelement <2 x i64> %a, i32 0
  %vecext1 = extractelement <2 x i64> %b, i32 0
  %xor.i = xor i64 %vecext1, %vecext
  %retval = bitcast i64 %xor.i to double
  ret double %retval
}
