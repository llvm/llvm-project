; RUN: llvm-as -disable-output <%s 2>&1 | FileCheck %s --implicit-check-not="attached to unexpected instruction kind"
;; Check that we allow intrinsics to have !DIAssignID attachments, but we do not
;; allow non-intrinsic calls to have them.
;; FIXME: Ideally we would also not allow non-store intrinsics, e.g. the
;; llvm.vp.load intrinsic in this test.

; CHECK: !DIAssignID attached to unexpected instruction kind
; CHECK-NEXT: call void @g()

declare void @g()

define void @f() !dbg !5 {
  call void @llvm.vp.store.v2i8.p0(<2 x i8> poison, ptr poison, <2 x i1> poison, i32 poison), !DIAssignID !6
  call void @llvm.vp.scatter.v2i8.v2p0(<2 x i8> poison, <2 x ptr> poison, <2 x i1> poison, i32 poison), !DIAssignID !7
  call void @llvm.experimental.vp.strided.store.v2i8.i64(<2 x i8> poison, ptr poison, i64 poison, <2 x i1> poison, i32 poison), !DIAssignID !8
  call void @llvm.masked.store.v2i8.p0(<2 x i8> poison, ptr poison, i32 1, <2 x i1> poison), !DIAssignID !9
  call void @llvm.masked.scatter.v2i8.v2p0(<2 x i8> poison, <2 x ptr> poison, i32 1, <2 x i1> poison), !DIAssignID !10
  %r = call <2 x i8> @llvm.vp.load.v2i8.p0(ptr poison, <2 x i1> poison, i32 poison), !DIAssignID !11
  call void @g(), !DIAssignID !12
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
!12 = distinct !DIAssignID()
