; Test to ensure that if an exact definition is imported, it is used in favor
; of an already-present declaration. Exact definitions from the same module may
; be optimized together. Thus care must be taken when importing only a subset
; of the definitions from a module (because other referenced definitions from
; that module may have been changed by the optimizer and may no longer match
; declarations already present in the module being imported into).

; Generate bitcode and index, and run the function import.
; `Inputs/attr_fixup_dae_noundef.ll` contains the post-"Dead Argument Elimination" IR, which
; removed the `noundef` from `@inner`.
; RUN: opt -module-summary %p/Inputs/attr_fixup_dae_noundef.ll -o %t.inputs.attr_fixup_dae_noundef.bc
; RUN: opt -module-summary %s -o %t.main.bc
; RUN: llvm-lto -thinlto -o %t.summary %t.main.bc %t.inputs.attr_fixup_dae_noundef.bc
; RUN: opt -passes=function-import -summary-file %t.summary.thinlto.bc %t.main.bc -S 2>&1 \
; RUN:   | FileCheck %s

define void @main()  {
  call void @outer(i32 noundef 1)
  call void @inner(i32 noundef 1)
  ; This call is UB due to signature mismatch with the actual definition.
  ; Make sure it does not lead to a crash.
  call void @inner2()
  ret void
}

; `@outer` should get imported.
; CHECK: define available_externally void @outer(i32 noundef %arg)
; CHECK-NEXT: call void @inner(i32 poison)
declare void @outer(i32 noundef)

; Because `@inner` is `noinline`, the definition should not be important.
; However, we should create a new declaration from the definition, which does
; not have the `noundef` attribute.
; CHECK: declare void @inner(i32)
declare void @inner(i32 noundef)

; CHECK: declare void @inner2(i32)
declare void @inner2()
