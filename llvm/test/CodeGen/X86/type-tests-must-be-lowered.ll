; RUN: not llc %s -mtriple=i686-- -O0 -filetype=null 2>&1 | FileCheck %s
; RUN: not llc %s -mtriple=x86_64-- -O0 -filetype=null 2>&1 | FileCheck %s

; The llvm.type.test, llvm.public.type.test, llvm.type.checked.load and
; llvm.type.checked.load.relative intrinsics are expected to be lowered by the
; LowerTypeTests pass before code generation. If one survives, emit a clean
; diagnostic instead of crashing (see issues #142937 and #164663).

; CHECK: type.test intrinsic must be lowered by the LowerTypeTests pass before code generation
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

; CHECK: public.type.test intrinsic must be lowered by the LowerTypeTests pass before code generation
define void @public_type_test() {
bb:
  %call = call i1 @llvm.public.type.test(ptr null, metadata !"typeinfo")
  br label %bb1

bb1:
  call void @llvm.assume(i1 %call)
  ret void
}

; CHECK: type.checked.load intrinsic must be lowered by the LowerTypeTests pass before code generation
define i1 @type_checked_load(ptr %vtable) {
  %pair = call { ptr, i1 } @llvm.type.checked.load(ptr %vtable, i32 4, metadata !"typeid")
  %ok = extractvalue { ptr, i1 } %pair, 1
  ret i1 %ok
}

; CHECK: type.checked.load.relative intrinsic must be lowered by the LowerTypeTests pass before code generation
define i1 @type_checked_load_relative(ptr %vtable) {
  %pair = call { ptr, i1 } @llvm.type.checked.load.relative(ptr %vtable, i32 4, metadata !"typeid")
  %ok = extractvalue { ptr, i1 } %pair, 1
  ret i1 %ok
}
