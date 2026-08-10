; RUN: not llvm-as -disable-output --verify-noalias-scope-decl-dom < %s 2>&1 | FileCheck %s

define void @test_single_scope01() nounwind ssp {
  tail call void @llvm.experimental.noalias.scope.decl(metadata !2)
  ret void
}

define void @test_single_scope02() nounwind ssp {
  tail call void @llvm.experimental.noalias.scope.decl(metadata !5)
  ret void
}
; CHECK: !id.scope.list must point to a list with a single scope
; CHECK-NEXT:   tail call void @llvm.experimental.noalias.scope.decl(metadata !5)

define void @test_single_scope03() nounwind ssp {
  tail call void @llvm.experimental.noalias.scope.decl(metadata !"test")
  ret void
}
; CHECK-NEXT: !id.scope.list must point to an MDNode
; CHECK-NEXT:   tail call void @llvm.experimental.noalias.scope.decl(metadata !"test")

define void @test_dom01() nounwind ssp {
  tail call void @llvm.experimental.noalias.scope.decl(metadata !2)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !8)
  ret void
}

define void @test_dom02() nounwind ssp {
  tail call void @llvm.experimental.noalias.scope.decl(metadata !2)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !6)
  ret void
}
; CHECK-NEXT: llvm.experimental.noalias.scope.decl dominates another one with the same scope
; CHECK-NEXT:   tail call void @llvm.experimental.noalias.scope.decl(metadata !2)

define void @test_dom03() nounwind ssp {
  tail call void @llvm.experimental.noalias.scope.decl(metadata !2)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !2)
  ret void
}
; CHECK-NEXT: llvm.experimental.noalias.scope.decl dominates another one with the same scope
; CHECK-NEXT:   tail call void @llvm.experimental.noalias.scope.decl(metadata !2)

; CHECK-NOT: llvm.experimental.noalias.scope.decl

; Function Attrs: inaccessiblememonly nounwind
declare void @llvm.experimental.noalias.scope.decl(metadata) #1

attributes #1 = { inaccessiblememonly nounwind }
!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang"}
!2 = !{!3}
!3 = distinct !{!3, !4, !"test: pA"}
!4 = distinct !{!4, !"test"}
!5 = !{!3, !3}
!6 = !{!3}
!7 = distinct !{!7, !4, !"test: pB"}
!8 = !{!7}
