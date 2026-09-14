; RUN: llc < %s -mtriple=x86_64 -stop-before=x86-cmov-conversion | FileCheck %s

; The unpredictable flag of a select must survive the CMOV rewrites in
; combineCMov, so that X86CmovConversion can see it. Every function below
; loses the flag without the fix.

; Simplified EFLAGS: the compare of the atomic result is folded into the
; flags of the lock add.
; CHECK-LABEL: name: simplified_eflags
; CHECK: unpredictable CMOV32rr
define i32 @simplified_eflags(ptr %p, i32 %x, i32 %y) {
entry:
  %old = atomicrmw add ptr %p, i32 1 seq_cst
  %c = icmp slt i32 %old, 0
  %r = select i1 %c, i32 %x, i32 %y, !unpredictable !0
  ret i32 %r
}

; and/or of two setcc sharing EFLAGS (oeq is ZF and not PF), folded into two
; CMOVs; both must keep the flag.
; CHECK-LABEL: name: double_cmov
; CHECK: unpredictable CMOV32rr
; CHECK: unpredictable CMOV32rr
define i32 @double_cmov(float %a, float %b, i32 %x, i32 %y) {
entry:
  %c = fcmp oeq float %a, %b
  %r = select i1 %c, i32 %x, i32 %y, !unpredictable !0
  ret i32 %r
}

; select (x == 0), 0, y -> select (x == 0), x, y.
; CHECK-LABEL: name: constant_to_register
; CHECK: unpredictable CMOV32rr
define i32 @constant_to_register(i32 %a, i32 %y) {
entry:
  %c = icmp eq i32 %a, 0
  %r = select i1 %c, i32 0, i32 %y, !unpredictable !0
  ret i32 %r
}

; select (x == 0), C, (cttz x) + C2 -> (cmov (C - C2), (cttz x)) + C2.
; CHECK-LABEL: name: cttz_add
; CHECK: unpredictable CMOV32rr
define i32 @cttz_add(i32 %a) {
entry:
  %t = call i32 @llvm.cttz.i32(i32 %a, i1 true)
  %add = add i32 %t, 1
  %c = icmp eq i32 %a, 0
  %r = select i1 %c, i32 33, i32 %add, !unpredictable !0
  ret i32 %r
}

declare i32 @llvm.cttz.i32(i32, i1)

!0 = !{}
