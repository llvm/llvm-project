; RUN: llvm-as -disable-output < %s 2>&1 | FileCheck --match-full-lines %s

define void @f() {
  ; CHECK: invalid llvm.dbg.def intrinsic lifetime
  call void @llvm.dbg.def(metadata i1 undef, metadata !3), !dbg !5
  ; CHECK: invalid llvm.dbg.def intrinsic referrer
  call void @llvm.dbg.def(metadata !3, metadata !3), !dbg !5
  ; CHECK: invalid llvm.dbg.kill intrinsic lifetime
  call void @llvm.dbg.kill(metadata i1 undef), !dbg !7
  ret void
}

define void @g() {
  call void @llvm.dbg.def(metadata !8, metadata i1 undef), !dbg !5
  ret void
}

define void @h() {
  ; CHECK: invalid llvm.dbg.def refers to an already-defined lifetime
  call void @llvm.dbg.def(metadata !8, metadata i1 undef), !dbg !5
  ret void
}

; CHECK: warning: ignoring invalid debug info in <stdin>

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
!8 = distinct !DILifetime(object: !4, location: !DIExpr())
