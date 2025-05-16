; RUN: llvm-reduce %s -o %t --abort-on-invalid-reduction --delta-passes=operands-to-args --test FileCheck --test-arg %s --test-arg --check-prefixes=INTERESTING --test-arg --input-file
; RUN: FileCheck %s --input-file %t --check-prefixes=RESULT

; Make sure there's no assert from trying to create a
; not-zeroinitializable target ext type


declare void @uses_ext_ty(target("sometarget.sometype"))
declare target("sometarget.sometype") @produces_ext_ty()

; INTERESTING: define void @not_zero_foldable(

; RESULT: define void @not_zero_foldable(target("sometarget.sometype") %call) {
; RESULT-NEXT: %call1 = call target("sometarget.sometype") @produces_ext_ty()
; RESULT-NEXT: call void @uses_ext_ty(target("sometarget.sometype") %call)
define void @not_zero_foldable() {
  %call = call target("sometarget.sometype") @produces_ext_ty()
  call void @uses_ext_ty(target("sometarget.sometype") %call)
  ret void
}

declare void @uses_zeroinit_ext_ty(target("spirv.zeroinit"))
declare target("sometarget.sometype") @produces_zeroinit_ext_ty()

; INTERESTING: define void @foldable_to_zero(
; RESULT: define void @foldable_to_zero(target("spirv.zeroinit") %call) {
define void @foldable_to_zero() {
  %call = call target("spirv.zeroinit") @produces_zeroinit_ext_ty()
  call void @uses_zeroinit_ext_ty(target("spirv.zeroinit") %call)
  ret void
}

