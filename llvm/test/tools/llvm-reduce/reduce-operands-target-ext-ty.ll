; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=operands-zero --test FileCheck --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefixes=CHECK,ZERO %s < %t

; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=operands-one --test FileCheck --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefixes=CHECK,ONE %s < %t

; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=operands-poison --test FileCheck --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefixes=CHECK,POISON %s < %t

declare void @uses_ext_ty(target("sometarget.sometype"))

; TODO: Should support reduce to poison
; CHECK-LABEL: @foo(
; ZERO: call void @uses_ext_ty(target("sometarget.sometype") %arg)
; ONE: call void @uses_ext_ty(target("sometarget.sometype") %arg)
; POISON: call void @uses_ext_ty(target("sometarget.sometype") poison)
define void @foo(target("sometarget.sometype") %arg) {
  call void @uses_ext_ty(target("sometarget.sometype") %arg)
  ret void
}

declare void @uses_zeroinit_ext_ty(target("sometarget.sometype"))

; CHECK-LABEL: @bar(
; ZERO: call void @uses_zeroinit_ext_ty(target("spirv.sometype") zeroinitializer)
; ONE: call void @uses_zeroinit_ext_ty(target("spirv.sometype") %arg)
; POISON: call void @uses_zeroinit_ext_ty(target("spirv.sometype") poison)
define void @bar(target("spirv.sometype") %arg) {
  call void @uses_zeroinit_ext_ty(target("spirv.sometype") %arg)
  ret void
}
