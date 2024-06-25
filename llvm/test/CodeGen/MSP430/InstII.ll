; RUN: llc -march=msp430 < %s | FileCheck %s
target datalayout = "e-p:16:8:8-i8:8:8-i16:8:8-i32:8:8"
target triple = "msp430-generic-generic"

define void @rra8m(ptr %i) {
entry:
; CHECK-LABEL: rra8m:
; CHECK: rra.b 2(r12)
  %0 = getelementptr inbounds i8, ptr %i, i16 2
  %1 = load i8, ptr %0, align 1
  %shr = ashr i8 %1, 1
  store i8 %shr, ptr %0, align 1
  ret void
}

define void @rra16m(ptr %i) {
entry:
; CHECK-LABEL: rra16m:
; CHECK: rra 4(r12)
  %0 = getelementptr inbounds i16, ptr %i, i16 2
  %1 = load i16, ptr %0, align 2
  %shr = ashr i16 %1, 1
  store i16 %shr, ptr %0, align 2
  ret void
}

; TODO: `clrc; rrc.b 2(r12)` is expected
define void @rrc8m(ptr %g) {
entry:
; CHECK-LABEL: rrc8m:
; CHECK: mov.b 2(r12), r13
; CHECK: clrc
; CHECK: rrc.b r13
; CHECK: mov.b r13, 2(r12)
  %add.ptr = getelementptr inbounds i8, ptr %g, i16 2
  %0 = load i8, ptr %add.ptr, align 1
  %1 = lshr i8 %0, 1
  store i8 %1, ptr %add.ptr, align 1
  ret void
}

; TODO: `clrc; rrc 4(r12)` is expected
define void @rrc16m(ptr %g) {
entry:
; CHECK-LABEL: rrc16m:
; CHECK: mov 4(r12), r13
; CHECK: clrc
; CHECK: rrc r13
; CHECK: mov r13, 4(r12)
  %add.ptr = getelementptr inbounds i16, ptr %g, i16 2
  %0 = load i16, ptr %add.ptr, align 2
  %shr = lshr i16 %0, 1
  store i16 %shr, ptr %add.ptr, align 2
  ret void
}

define void @sxt16m(ptr %x) {
entry:
; CHECK-LABEL: sxt16m:
; CHECK: sxt 4(r12)
  %add.ptr = getelementptr inbounds i16, ptr %x, i16 2
  %0 = bitcast ptr %add.ptr to ptr
  %1 = load i8, ptr %0, align 1
  %conv = sext i8 %1 to i16
  store i16 %conv, ptr %add.ptr, align 2
  ret void
}

