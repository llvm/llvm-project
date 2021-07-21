; RUN: llvm-as -disable-output < %s 2>&1 | FileCheck --match-full-lines %s

; CHECK: MDNode incompatible with Debug Info Version
; CHECK-NEXT: !DIExpression()
; CHECK-NEXT: 4
; CHECK-NEXT: debug intrinsic incompatible with Debug Info Version
; CHECK-NEXT:   call void @llvm.dbg.declare(metadata i1 undef, metadata !3, metadata !DIExpression()), !dbg !5
; CHECK-NEXT: 4
; CHECK-NEXT: debug intrinsic incompatible with Debug Info Version
; CHECK-NEXT:   call void @llvm.dbg.value(metadata i1 undef, metadata !3, metadata !DIExpression()), !dbg !5
; CHECK-NEXT: 4
; CHECK-NEXT: warning: ignoring invalid debug info in <stdin>

define void @f() {
  call void @llvm.dbg.declare(metadata i1 undef, metadata !4, metadata !3), !dbg !5
  call void @llvm.dbg.value(metadata i1 undef, i64 0, metadata !4, metadata !3), !dbg !5
  ret void
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #0

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #0

attributes #0 = { nofree nosync nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug)
!1 = !DIFile(filename: "<stdin>", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 4}
!3 = !DIExpression()
!4 = distinct !DILocalVariable(name: "foo", scope: !6)
!5 = !DILocation(line: 4, column: 1, scope: !6)
!6 = distinct !DISubprogram(scope: null, spFlags: DISPFlagDefinition, unit: !0)
!7 = !DILocation(line: 8, column: 1, scope: !6)
