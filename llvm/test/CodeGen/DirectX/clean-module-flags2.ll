; RUN: opt -S -dxil-prepare < %s | FileCheck %s

; CHECK: define void @main()
; Make sure non-dxil module flags are removed.
; CHECK-NOT:"wchar_size"
; CHECK-NOT:"frame-pointer"

; Function Attrs: nounwind memory(none)
define void @main() local_unnamed_addr #0 {
entry:
  ret void
}

attributes #0 = { nounwind memory(none) }
!llvm.module.flags = !{!0, !1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
