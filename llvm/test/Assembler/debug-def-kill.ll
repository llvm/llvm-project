; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck --match-full-lines %s
; RUN: verify-uselistorder %s

define void @f() {
; CHECK: call void @llvm.dbg.def(metadata !3, metadata i1 undef), !dbg !5
  call void @llvm.dbg.def(metadata !3, metadata i1 undef), !dbg !5
; CHECK: call void @llvm.dbg.kill(metadata !3), !dbg !7
  call void @llvm.dbg.kill(metadata !3), !dbg !7
  ret void
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.def(metadata, metadata) #0

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.kill(metadata) #0

attributes #0 = { nofree nosync nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug)
!1 = !DIFile(filename: "<stdin>", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 4}
!3 = distinct !DILifetime(object: !4, location: !DIExpr())
!4 = distinct !DIFragment()
!5 = !DILocation(line: 4, column: 1, scope: !6)
!6 = distinct !DISubprogram(scope: null, spFlags: DISPFlagDefinition, unit: !0)
!7 = !DILocation(line: 8, column: 1, scope: !6)
