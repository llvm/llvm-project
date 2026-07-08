; RUN: not llc %s -mtriple=i686-- -O0 -filetype=null 2>&1 | FileCheck %s
; RUN: not llc %s -mtriple=x86_64-- -O0 -filetype=null 2>&1 | FileCheck %s

; llvm.type.checked.load is expected to be lowered by the LowerTypeTests
; pass before code generation.
;
; If it survives, emit a clean diagnostic instead of crashing (see issue #164663).

; CHECK: llvm.type.checked.load intrinsic must be lowered by the LowerTypeTests pass

define i1 @type_checked_load(ptr %vtable) {
  %pair = call { ptr, i1 } @llvm.type.checked.load(ptr %vtable, i32 4, metadata !"typeid")
  %ok = extractvalue { ptr, i1 } %pair, 1
  ret i1 %ok
}
