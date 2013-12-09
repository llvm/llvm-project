; RUN: llc < %s -mtriple=sparc64-pc-openbsd -disable-sparc-leaf-proc | FileCheck %s
; Testing 64-bit conditionals. The sparc64 triple is an alias for sparcv9.

; CHECK: cmpri
; CHECK: cmp %i1, 1
; CHECK: be %xcc,
define void @cmpri(i64* %p, i64 %x) {
entry:
  %tobool = icmp eq i64 %x, 1
  br i1 %tobool, label %if.end, label %if.then

if.then:
  store i64 %x, i64* %p, align 8
  br label %if.end

if.end:
  ret void
}

; CHECK: cmprr
; CHECK: cmp %i1, %i2
; CHECK: bgu %xcc,
define void @cmprr(i64* %p, i64 %x, i64 %y) {
entry:
  %tobool = icmp ugt i64 %x, %y
  br i1 %tobool, label %if.end, label %if.then

if.then:
  store i64 %x, i64* %p, align 8
  br label %if.end

if.end:
  ret void
}

; CHECK: selecti32_xcc
; CHECK: cmp %i0, %i1
; CHECK: movg %xcc, %i2, %i3
; CHECK: restore %g0, %i3, %o0
define i32 @selecti32_xcc(i64 %x, i64 %y, i32 %a, i32 %b) {
entry:
  %tobool = icmp sgt i64 %x, %y
  %rv = select i1 %tobool, i32 %a, i32 %b
  ret i32 %rv
}

; CHECK: selecti64_xcc
; CHECK: cmp %i0, %i1
; CHECK: movg %xcc, %i2, %i3
; CHECK: restore %g0, %i3, %o0
define i64 @selecti64_xcc(i64 %x, i64 %y, i64 %a, i64 %b) {
entry:
  %tobool = icmp sgt i64 %x, %y
  %rv = select i1 %tobool, i64 %a, i64 %b
  ret i64 %rv
}

; CHECK: selecti64_icc
; CHECK: cmp %i0, %i1
; CHECK: movg %icc, %i2, %i3
; CHECK: restore %g0, %i3, %o0
define i64 @selecti64_icc(i32 %x, i32 %y, i64 %a, i64 %b) {
entry:
  %tobool = icmp sgt i32 %x, %y
  %rv = select i1 %tobool, i64 %a, i64 %b
  ret i64 %rv
}

; CHECK: selecti64_fcc
; CHECK: fcmps %f1, %f3
; CHECK: movul %fcc0, %i2, %i3
; CHECK: restore %g0, %i3, %o0
define i64 @selecti64_fcc(float %x, float %y, i64 %a, i64 %b) {
entry:
  %tobool = fcmp ult float %x, %y
  %rv = select i1 %tobool, i64 %a, i64 %b
  ret i64 %rv
}

; CHECK: selectf32_xcc
; CHECK: cmp %i0, %i1
; CHECK: fmovsg %xcc, %f5, %f7
; CHECK: fmovs %f7, %f1
define float @selectf32_xcc(i64 %x, i64 %y, float %a, float %b) {
entry:
  %tobool = icmp sgt i64 %x, %y
  %rv = select i1 %tobool, float %a, float %b
  ret float %rv
}

; CHECK: selectf64_xcc
; CHECK: cmp %i0, %i1
; CHECK: fmovdg %xcc, %f4, %f6
; CHECK: fmovd %f6, %f0
define double @selectf64_xcc(i64 %x, i64 %y, double %a, double %b) {
entry:
  %tobool = icmp sgt i64 %x, %y
  %rv = select i1 %tobool, double %a, double %b
  ret double %rv
}

; The MOVXCC instruction can't use %g0 for its tied operand.
; CHECK: select_consti64_xcc
; CHECK: cmp
; CHECK: movg %xcc, 123, %i{{[0-2]}}
define i64 @select_consti64_xcc(i64 %x, i64 %y) {
entry:
  %tobool = icmp sgt i64 %x, %y
  %rv = select i1 %tobool, i64 123, i64 0
  ret i64 %rv
}

; CHECK-LABEL: setcc_resultty
; CHECK:       cmp
; CHECK:       movne %xcc, 1, [[R:%[gilo][0-7]]]
; CHECK:       or [[R]], %i1, %i0

define i1 @setcc_resultty(i64 %a, i1 %b) {
  %a0 = tail call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %a, i64 32)
  %a1 = extractvalue { i64, i1 } %a0, 1
  %a4 = or i1 %a1, %b
  ret i1 %a4
}

declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64)
