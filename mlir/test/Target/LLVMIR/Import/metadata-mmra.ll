; RUN: mlir-translate -import-llvm -split-input-file %s | FileCheck %s

; CHECK-DAG: #[[$MMRA0:.+]] = #llvm.mmra_tag<"foo":"bar">
; CHECK-DAG: #[[$MMRA1:.+]] = #llvm.mmra_tag<"amdgpu-synchronize-as":"local">

; CHECK-LABEL: llvm.func @native
define void @native(ptr %x, ptr %y) {
  ; CHECK: llvm.load
  ; CHECK-SAME: llvm.mmra = #[[$MMRA0]]
  %v = load i32, ptr %x, align 4, !mmra !0
  ; CHECK: llvm.fence
  ; CHECK-SAME: llvm.mmra = [#[[$MMRA1]], #[[$MMRA0]]]
  fence syncscope("workgroup-one-as") release, !mmra !2
  ; CHECK: llvm.store {{.*}}, !llvm.ptr{{$}}
  store i32 %v, ptr %y, align 4, !mmra !3
  ret void
}

!0 = !{!"foo", !"bar"}
!1 = !{!"amdgpu-synchronize-as", !"local"}
!2 = !{!1, !0}
!3 = !{}
