; RUN: llvm-as -disable-output <%s 2>&1 | FileCheck %s --implicit-check-not="attached to unexpected instruction kind"
;; Check that we allow vector store intrinsics to have !DIAssignID attachments,
;; but we do not allow non-store intrinsics to have them.

; CHECK: !DIAssignID attached to unexpected instruction kind
; CHECK-NEXT: @llvm.vp.load.v2i8.p0

define void @f() !dbg !5 {
  call void @llvm.vp.store.v2i8.p0(<2 x i8> undef, ptr undef, <2 x i1> undef, i32 undef), !DIAssignID !6
  call void @llvm.vp.scatter.v2i8.v2p0(<2 x i8> undef, <2 x ptr> undef, <2 x i1> undef, i32 undef), !DIAssignID !7
  call void @llvm.experimental.vp.strided.store.v2i8.i64(<2 x i8> undef, ptr undef, i64 undef, <2 x i1> undef, i32 undef), !DIAssignID !8
  call void @llvm.masked.store.v2i8.p0(<2 x i8> undef, ptr undef, i32 1, <2 x i1> undef), !DIAssignID !9
  call void @llvm.masked.scatter.v2i8.v2p0(<2 x i8> undef, <2 x ptr> undef, i32 1, <2 x i1> undef), !DIAssignID !10
  %r = call <2 x i8> @llvm.vp.load.v2i8.p0(ptr undef, <2 x i1> undef, i32 undef), !DIAssignID !11
  ret void
}

!llvm.module.flags = !{!0}
!llvm.dbg.cu = !{!1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DICompileUnit(language: DW_LANG_Swift, producer: "clang",
                             file: !2, emissionKind: 2)
!2 = !DIFile(filename: "path/to/file", directory: "/path/to/dir")
!3 = !{null}
!4 = !DISubroutineType(types: !3)
!5 = distinct !DISubprogram(name: "f", scope: !2, file: !2, line: 1, type: !4, scopeLine: 2, unit: !1)
!6 = distinct !DIAssignID()
!7 = distinct !DIAssignID()
!8 = distinct !DIAssignID()
!9 = distinct !DIAssignID()
!10 = distinct !DIAssignID()
!11 = distinct !DIAssignID()
