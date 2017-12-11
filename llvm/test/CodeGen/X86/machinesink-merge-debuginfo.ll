; RUN: llc -simplify-mir -stop-after=machine-sink < %s -o - | FileCheck %s

; ModuleID = 'test-sink-debug.cpp'
source_filename = "test-sink-debug.cpp"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind readnone uwtable
define double @_Z3fooddb(double %x, double %y, i1 zeroext %c) local_unnamed_addr !dbg !7 {
  tail call void @llvm.dbg.value(metadata double %x, metadata !13, metadata !DIExpression()), !dbg !16
  tail call void @llvm.dbg.value(metadata double %y, metadata !14, metadata !DIExpression()), !dbg !16
  tail call void @llvm.dbg.value(metadata i1 %c, metadata !15, metadata !DIExpression()), !dbg !16
  %a = fdiv double %x, 3.000000e+00
  %b = fdiv double %y, 5.000000e+00, !dbg !17
  br i1 %c, label %first, label %second
first:
  %e = fadd double %a, 1.000000e+00
  br label %final
second:
  %f = fadd double %b, 1.000000e+00, !dbg !17
; CHECK:  debug-location !17
; CHECK:  debug-location !17
  br label %final
final:
  %cond = phi double [%e, %first], [%f, %second]
  %d = fadd double %cond, 1.000000e+00
  ret double %d
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #1 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 6.0.0 (trunk 313291)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "test-sink-debug.cpp", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 6.0.0 (trunk 313291)"}
!7 = distinct !DISubprogram(name: "foo", linkageName: "_Z3fooddb", scope: !1, file: !1, line: 1, type: !8, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !12)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !10, !10, !11}
!10 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
!11 = !DIBasicType(name: "bool", size: 8, encoding: DW_ATE_boolean)
!12 = !{!13, !14, !15}
!13 = !DILocalVariable(name: "x", arg: 1, scope: !7, file: !1, line: 1, type: !10)
!14 = !DILocalVariable(name: "y", arg: 2, scope: !7, file: !1, line: 1, type: !10)
!15 = !DILocalVariable(name: "c", arg: 3, scope: !7, file: !1, line: 1, type: !11)
!16 = !DILocation(line: 1, column: 19, scope: !7)
!17 = !DILocation(line: 2, column: 26, scope: !7)
