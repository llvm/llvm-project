; RUN: not llc %s -mtriple=i686-- -O0 -filetype=null 2>&1 | FileCheck %s
; RUN: not llc %s -mtriple=x86_64-- -O0 -filetype=null 2>&1 | FileCheck %s

; llvm.type.test is expected to be lowered by the LowerTypeTests
; pass before code generation.
;
; If it survives, emit a clean diagnostic instead of crashing (see issue #142937).

; CHECK: must be lowered by the LowerTypeTests pass

define void @type_test() {
bb:
  %call = tail call i1 @llvm.type.test(ptr null, metadata !"typeinfo")
  br i1 %call, label %bb2, label %bb1

bb1:
  tail call void @llvm.ubsantrap(i8 2)
  unreachable

bb2:
  ret void
}
