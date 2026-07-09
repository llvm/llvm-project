; RUN: not llc %s -mtriple=i686-- -O0 -filetype=null 2>&1 | FileCheck %s
; RUN: not llc %s -mtriple=x86_64-- -O0 -filetype=null 2>&1 | FileCheck %s

; llvm.public.type.test is expected to be lowered by the LowerTypeTests
; pass before code generation.
;
; If it survives, emit a clean diagnostic instead of crashing (see issue #142937).

; CHECK: must be lowered by the LowerTypeTests pass

define void @public_type_test() {
bb:
  %call = call i1 @llvm.public.type.test(ptr null, metadata !"typeinfo")
  br label %bb1

bb1:
  call void @llvm.assume(i1 %call)
  ret void
}
