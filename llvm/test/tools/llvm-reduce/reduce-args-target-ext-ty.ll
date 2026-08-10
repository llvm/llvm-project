; RUN: llvm-reduce %s -o %t --abort-on-invalid-reduction --delta-passes=arguments --test FileCheck --test-arg %s --test-arg --check-prefixes=INTERESTING --test-arg --input-file
; RUN: FileCheck %s --input-file %t --check-prefixes=RESULT

declare void @uses_ext_ty(target("sometarget.sometype"))
declare target("sometarget.sometype") @produces_ext_ty()

; INTERESTING: @interesting(
; RESULT: @interesting(
; RESULT: void @uses_ext_ty()
define void @interesting(target("sometarget.sometype") %arg) {
  call void @uses_ext_ty(target("sometarget.sometype") %arg)
  ret void
}
