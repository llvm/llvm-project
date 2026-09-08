; RUN: llc -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck %s
; RUN: llc -mtriple=i686-unknown-linux-gnu < %s | FileCheck %s --check-prefix=CHECK32

; A function with the throws (herbception) attribute returns its value with a
; discriminant. On x86-64/x86-32 the discriminant is carried in the carry flag
; (CF) instead of a return register.

; Success: discriminant is false (0) -> CF is clear.
define { i64, i1 } @ret_success(i64 %x) #0 {
; CHECK-LABEL: ret_success:
; CHECK:       # %bb.0:
; CHECK-NEXT:    movq %rdi, %rax
; CHECK-NEXT:    clc
; CHECK-NEXT:    retq
entry:
  %r.i = insertvalue { i64, i1 } poison, i64 %x, 0
  %r = insertvalue { i64, i1 } %r.i, i1 false, 1
  ret { i64, i1 } %r
}

; Error: discriminant is true (1) -> CF is set.
define { i64, i1 } @ret_error(i64 %x) #0 {
; CHECK-LABEL: ret_error:
; CHECK:       # %bb.0:
; CHECK-NEXT:    movq %rdi, %rax
; CHECK-NEXT:    stc
; CHECK-NEXT:    retq
entry:
  %r.i = insertvalue { i64, i1 } poison, i64 %x, 0
  %r = insertvalue { i64, i1 } %r.i, i1 true, 1
  ret { i64, i1 } %r
}

; The caller reads the discriminant from CF right after the call (setb).
define i64 @call_and_select(i64 %x) #1 {
; CHECK-LABEL: call_and_select:
; CHECK:       # %bb.0:
; CHECK-NEXT:    pushq %rax
; CHECK-NEXT:    callq ret_error@PLT
; CHECK-NEXT:    movl $100, %ecx
; CHECK-NEXT:    cmovbq %rcx, %rax
; CHECK-NEXT:    popq %rcx
; CHECK-NEXT:    retq
; CHECK32-LABEL: call_and_select:
; CHECK32:       # %bb.0:
; CHECK32:         calll ret_error@PLT
; CHECK32-NOT:     setb
; CHECK32-NOT:     testb
; CHECK32:         jae .LBB2_2
; CHECK32-NEXT:  # %bb.1:
; CHECK32-NEXT:    xorl %edx, %edx
; CHECK32-NEXT:    movl $100, %eax
; CHECK32-NEXT:  .LBB2_2: # %entry
; CHECK32-NEXT:    addl $12, %esp
; CHECK32-NEXT:    retl
entry:
  %c = call { i64, i1 } @ret_error(i64 %x)
  %val = extractvalue { i64, i1 } %c, 0
  %disc = extractvalue { i64, i1 } %c, 1
  %sel = select i1 %disc, i64 100, i64 %val
  ret i64 %sel
}

attributes #0 = { throws }
attributes #1 = { nounwind }

; Herbception (throws): branch on carry flag. When the discriminant is used
; only for a branch, the backend folds setb + test + jcc into a single jcc
; on CF (jae/jb), eliminating the setb.
declare void @capture(i32) #3
define void @call_and_branch() #1 {
; CHECK-LABEL: call_and_branch:
; CHECK:       # %bb.0:
; CHECK-NEXT:    pushq %rax
; CHECK-NEXT:    callq ret_error@PLT
; CHECK-NEXT:    jae .LBB3_1
; CHECK-NEXT:  # %bb.2: # %err
; CHECK-NEXT:    movl $7, %edi
; CHECK-NEXT:    popq %rax
; CHECK-NEXT:    jmp capture@PLT # TAILCALL
; CHECK-NEXT:  .LBB3_1: # %cont
; CHECK-NEXT:    popq %rax
; CHECK-NEXT:    retq
; CHECK32-LABEL: call_and_branch:
; CHECK32:       # %bb.0:
; CHECK32:         calll ret_error@PLT
; CHECK32-NEXT:    jae .LBB3_2
; CHECK32-NEXT:  # %bb.1: # %err
; CHECK32-NEXT:    movl $7, (%esp)
; CHECK32-NEXT:    calll capture@PLT
; CHECK32-NEXT:  .LBB3_2: # %cont
; CHECK32-NEXT:    addl $12, %esp
; CHECK32-NEXT:    retl
entry:
  %call = tail call { i32, i1 } @ret_error()
  %d = extractvalue { i32, i1 } %call, 1
  br i1 %d, label %err, label %cont
err:
  tail call void @capture(i32 7)
  ret void
cont:
  ret void
}

attributes #3 = { nounwind }
