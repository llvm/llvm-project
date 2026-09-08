; RUN: llc -mtriple=x86_64-unknown-windows-msvc < %s | FileCheck %s
; RUN: llc -mtriple=i686-unknown-windows-msvc < %s | FileCheck %s --check-prefix=CHECK32
; RUN: llc -mtriple=aarch64-unknown-windows-msvc < %s | FileCheck %s --check-prefix=CHECK64

; Herbception (throws): on Win64, a throws call with shadow space emits
; ADJCALLSTACKUP64 32,0 between the call and the branch. The HERB_SETCCr
; has already captured CF into a register, so ADJCALLSTACKUP64's pessimistic
; EFLAGS def must not block the setb/test/jcc -> jcc fold.

; Branch-only discriminant use: folds to a single jcc on CF (no setb/testb).
declare void @capture(i32)
declare { i32, i1 } @foo(i32) #0

define void @call_and_branch(i32 %x) #1 {
; CHECK-LABEL: call_and_branch:
; CHECK:       callq foo
; CHECK-NOT:   setb
; CHECK-NOT:   testb
; CHECK:       j{{b|ae}} .LBB0_
; CHECK32-LABEL: _call_and_branch:
; CHECK32:       calll _foo
; CHECK32-NOT:   setb
; CHECK32-NOT:   testb
; CHECK32:       j{{b|ae}} LBB0_
; CHECK64-LABEL: call_and_branch:
; CHECK64:       bl foo
; CHECK64-NEXT:  b.lo .LBB0_
entry:
  %c = call { i32, i1 } @foo(i32 %x) #1
  %d = extractvalue { i32, i1 } %c, 1
  br i1 %d, label %err, label %cont
err:
  call void @capture(i32 7)
  ret void
cont:
  ret void
}

; Callee side: throws function returns discriminant in CF.
; Success path: clc + ret.
define { i32, i1 } @ret_success(i32 %x) #0 {
; CHECK-LABEL: ret_success:
; CHECK:       movl %ecx, %eax
; CHECK-NEXT:  clc
; CHECK:       retq
; CHECK32-LABEL: _ret_success:
; CHECK32:       movl 4(%esp), %eax
; CHECK32-NEXT:  clc
; CHECK32:       retl
; CHECK64-LABEL: ret_success:
; CHECK64:       mov w8, wzr
; CHECK64-NEXT:  cmp w8, #1
; CHECK64:       ret
entry:
  %r = insertvalue { i32, i1 } poison, i32 %x, 0
  %r1 = insertvalue { i32, i1 } %r, i1 false, 1
  ret { i32, i1 } %r1
}

; Error path: stc + ret.
define { i32, i1 } @ret_error(i32 %x) #0 {
; CHECK-LABEL: ret_error:
; CHECK:       movl %ecx, %eax
; CHECK-NEXT:  stc
; CHECK:       retq
; CHECK32-LABEL: _ret_error:
; CHECK32:       movl 4(%esp), %eax
; CHECK32-NEXT:  stc
; CHECK32:       retl
; CHECK64-LABEL: ret_error:
; CHECK64:       mov w8, #1
; CHECK64-NEXT:  cmp w8, #1
; CHECK64:       ret
entry:
  %r = insertvalue { i32, i1 } poison, i32 %x, 0
  %r1 = insertvalue { i32, i1 } %r, i1 true, 1
  ret { i32, i1 } %r1
}

; CMOV discriminant use: folds setb/testb/cmovne into cmovb/cmovae on
; x86_64, and setb/testb/jne into jb/jae on i686 (where select lowers to
; a branch with an intervening MOV32r0 that the peephole now skips).
define i64 @call_and_select(i64 %x) #1 {
; CHECK-LABEL: call_and_select:
; CHECK:       callq foo
; CHECK-NOT:   setb
; CHECK-NOT:   testb
; CHECK:       cmov{{b|ae}}
; CHECK32-LABEL: _call_and_select:
; CHECK32:       calll _foo
; CHECK32-NOT:   setb
; CHECK32-NOT:   testb
; CHECK32:       j{{b|ae}} LBB3_
entry:
  %c = call { i64, i1 } @foo(i64 %x) #1
  %val = extractvalue { i64, i1 } %c, 0
  %disc = extractvalue { i64, i1 } %c, 1
  %sel = select i1 %disc, i64 100, i64 %val
  ret i64 %sel
}

attributes #0 = { throws }
attributes #1 = { throws }
