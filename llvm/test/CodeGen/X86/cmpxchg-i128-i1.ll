; RUN: llc -mcpu=core-avx2 -mtriple=x86_64 -o - %s | FileCheck %s

define i1 @try_cmpxchg(i128* %addr, i128 %desired, i128 %new) {
; CHECK-LABEL: try_cmpxchg:
; CHECK: cmpxchg16b
; CHECK-NOT: cmp
; CHECK: sete %al
; CHECK: retq
  %pair = cmpxchg i128* %addr, i128 %desired, i128 %new seq_cst seq_cst
  %success = extractvalue { i128, i1 } %pair, 1
  ret i1 %success
}

define void @cmpxchg_flow(i128* %addr, i128 %desired, i128 %new) {
; CHECK-LABEL: cmpxchg_flow:
; CHECK: cmpxchg16b
; CHECK-NOT: cmp
; CHECK-NOT: set
; CHECK: {{jne|jeq}}
  %pair = cmpxchg i128* %addr, i128 %desired, i128 %new seq_cst seq_cst
  %success = extractvalue { i128, i1 } %pair, 1
  br i1 %success, label %true, label %false

true:
  call void @foo()
  ret void

false:
  call void @bar()
  ret void
}

; Can't use the flags here because cmpxchg16b only sets ZF.
define i1 @cmpxchg_arithcmp(i128* %addr, i128 %desired, i128 %new) {
; CHECK-LABEL: cmpxchg_arithcmp:
; CHECK: cmpxchg16b
; CHECK: cmpq
; CHECK: retq
  %pair = cmpxchg i128* %addr, i128 %desired, i128 %new seq_cst seq_cst
  %oldval = extractvalue { i128, i1 } %pair, 0
  %success = icmp sge i128 %oldval, %desired
  ret i1 %success
}

define i128 @cmpxchg_zext(i128* %addr, i128 %desired, i128 %new) {
; CHECK-LABEL: cmpxchg_zext:
; CHECK: xorl
; CHECK: cmpxchg16b
; CHECK-NOT: cmpq
; CHECK: sete
  %pair = cmpxchg i128* %addr, i128 %desired, i128 %new seq_cst seq_cst
  %success = extractvalue { i128, i1 } %pair, 1
  %mask = zext i1 %success to i128
  ret i128 %mask
}


define i128 @cmpxchg_use_eflags_and_val(i128* %addr, i128 %offset) {
; CHECK-LABEL: cmpxchg_use_eflags_and_val:

; CHECK: cmpxchg16b
; CHECK-NOT: cmpq
; CHECK: jne
entry:
  %init = load atomic i128, i128* %addr seq_cst, align 16
  br label %loop

loop:
  %old = phi i128 [%init, %entry], [%oldval, %loop]
  %new = add i128 %old, %offset

  %pair = cmpxchg i128* %addr, i128 %old, i128 %new seq_cst seq_cst
  %oldval = extractvalue { i128, i1 } %pair, 0
  %success = extractvalue { i128, i1 } %pair, 1

  br i1 %success, label %done, label %loop

done:
  ret i128 %old
}

declare void @foo()
declare void @bar()
