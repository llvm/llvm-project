; RUN: llc -O0 -fast-isel -stop-after=finalize-isel < %s | FileCheck %s
target triple = "x86_64-unknown-linux-gnu"

define void @undef() {
entry:
; CHECK-LABEL: name: undef
; CHECK: DBG_DEF !3, $noreg, debug-location !5
; CHECK: DBG_KILL !3, debug-location !7
  call void @llvm.dbg.def(metadata !3, metadata i1 undef), !dbg !5
  call void @llvm.dbg.kill(metadata !3), !dbg !7
  unreachable
}

define void @i1() {
entry:
; CHECK-LABEL: name: i1
; CHECK: DBG_DEF !3, i1 true, debug-location !5
; CHECK: DBG_KILL !3, debug-location !7
  call void @llvm.dbg.def(metadata !3, metadata i1 true), !dbg !5
  call void @llvm.dbg.kill(metadata !3), !dbg !7
  unreachable
}

define void @i128() {
entry:
; CHECK-LABEL: name: i128
; CHECK: DBG_DEF !3, i128 36893488147419103232, debug-location !5
; CHECK: DBG_KILL !3, debug-location !7
  call void @llvm.dbg.def(metadata !3, metadata i128 36893488147419103232), !dbg !5
  call void @llvm.dbg.kill(metadata !3), !dbg !7
  unreachable
}

define void @float() {
entry:
; CHECK-LABEL: name: float
; CHECK: DBG_DEF !3, float 1.000000e+00, debug-location !5
; CHECK: DBG_KILL !3, debug-location !7
  call void @llvm.dbg.def(metadata !3, metadata float 1.000000e+00), !dbg !5
  call void @llvm.dbg.kill(metadata !3), !dbg !7
  unreachable
}

define void @alloca() {
entry:
; CHECK-LABEL: name: alloca
; CHECK: DBG_DEF !3, %stack.{{[0-9]+}}, debug-location !5
; CHECK: DBG_KILL !3, debug-location !7
  %0 = alloca i1
  call void @llvm.dbg.def(metadata !3, metadata i1* %0), !dbg !5
  call void @llvm.dbg.kill(metadata !3), !dbg !7
  unreachable
}

define void @argument(i32 %0) {
entry:
; CHECK-LABEL: name: argument
; CHECK: DBG_DEF !3, %{{[0-9]+}}, debug-location !5
; CHECK: DBG_KILL !3, debug-location !7
  call void @llvm.dbg.def(metadata !3, metadata i32 %0), !dbg !5
  call void @llvm.dbg.kill(metadata !3), !dbg !7
  unreachable
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.def(metadata, metadata) #0

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.kill(metadata) #0

attributes #0 = { nofree nosync nounwind readnone speculatable willreturn }

!llvm.module.flags = !{!2}
!llvm.dbg.cu = !{!0}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug)
!1 = !DIFile(filename: "<stdin>", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DILifetime(object: !4, location: !DIExpr())
!4 = distinct !DIFragment()
!5 = !DILocation(line: 4, column: 1, scope: !6)
!6 = distinct !DISubprogram(scope: null, spFlags: DISPFlagDefinition, unit: !0)
!7 = !DILocation(line: 8, column: 1, scope: !6)
