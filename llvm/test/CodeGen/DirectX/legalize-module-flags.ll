; RUN: opt -S -dxil-prepare < %s | FileCheck %s

; Make sure behavior flag > 6 is fixed.
; CHECK: !{i32 2, !"frame-pointer", i32 2}
; CHECK: !{i32 2, !"Dwarf Version", i32 4}
; CHECK: !{i32 2, !"Debug Info Version", i32 3}

; Function Attrs: nounwind memory(none)
define void @main() local_unnamed_addr #0 {
entry:
  ret void
}

attributes #0 = { nounwind memory(none) }
!llvm.module.flags = !{!0, !1, !2, !3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{i32 7, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
