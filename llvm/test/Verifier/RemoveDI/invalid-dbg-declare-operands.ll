; RUN: llvm-as %s -o - 2>&1 | FileCheck %s
; CHECK: invalid #dbg record expression
;
; Fossilised debug-info with only two arguments to dbg.declare have been
; spotted in LLVMs test suite (debug-info-always-inline.ll), test that this
; does not cause a crash. LLVM needs to be able to autoupgrade invalid
; dbg.declares to invalid #dbg_declares because this occurs before the
; Verifier runs.

; ModuleID = 'out.ll'
source_filename = "llvm/test/DebugInfo/Generic/debug-info-always-inline.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @llvm.dbg.declare(metadata, metadata, metadata)

; Function Attrs: alwaysinline nounwind sspstrong
define i32 @_Z3foov() #0 !dbg !7 {
entry:
  %arr = alloca [10 x i32], align 16, !dbg !10
  %sum = alloca i32, align 4, !dbg !11
  call void @llvm.dbg.declare(metadata ptr %sum,  metadata !26), !dbg !11
  store i32 5, ptr %arr, align 4, !dbg !12
  store i32 4, ptr %sum, align 4, !dbg !13
  %0 = load i32, ptr %sum, align 4, !dbg !14
  ret i32 %0, !dbg !15
}

; Function Attrs: nounwind sspstrong
define i32 @main() #1 !dbg !16 {
entry:
  %retval = alloca i32, align 4, !dbg !17
  %i = alloca i32, align 4, !dbg !18
  store i32 0, ptr %retval, align 4, !dbg !19
  call void @_Z3barv(), !dbg !20
  %call = call i32 @_Z3foov(), !dbg !21
  store i32 %call, ptr %i, align 4, !dbg !22
  %0 = load i32, ptr %i, align 4, !dbg !23
  ret i32 %0, !dbg !24
}

declare void @_Z3barv() #2

attributes #0 = { alwaysinline nounwind sspstrong "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind sspstrong "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}
!llvm.dbg.cu = !{!3}
!llvm.debugify = !{!5, !6}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = !{!"clang version 3.6.0 (217844)"}
!3 = distinct !DICompileUnit(language: DW_LANG_C, file: !4, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!4 = !DIFile(filename: "/fast/fs/llvm-main/llvm/test/DebugInfo/Generic/debug-info-always-inline.ll", directory: "/")
!5 = !{i32 14}
!6 = !{i32 7}
!7 = distinct !DISubprogram(name: "_Z3foov", linkageName: "_Z3foov", scope: null, file: !4, line: 1, type: !8, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !3, retainedNodes: !9)
!8 = !DISubroutineType(types: !9)
!9 = !{}
!10 = !DILocation(line: 1, column: 1, scope: !7)
!11 = !DILocation(line: 2, column: 1, scope: !7)
!12 = !DILocation(line: 3, column: 1, scope: !7)
!13 = !DILocation(line: 4, column: 1, scope: !7)
!14 = !DILocation(line: 5, column: 1, scope: !7)
!15 = !DILocation(line: 6, column: 1, scope: !7)
!16 = distinct !DISubprogram(name: "main", linkageName: "main", scope: null, file: !4, line: 7, type: !8, scopeLine: 7, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !3, retainedNodes: !9)
!17 = !DILocation(line: 7, column: 1, scope: !16)
!18 = !DILocation(line: 8, column: 1, scope: !16)
!19 = !DILocation(line: 9, column: 1, scope: !16)
!20 = !DILocation(line: 10, column: 1, scope: !16)
!21 = !DILocation(line: 11, column: 1, scope: !16)
!22 = !DILocation(line: 12, column: 1, scope: !16)
!23 = !DILocation(line: 13, column: 1, scope: !16)
!24 = !DILocation(line: 14, column: 1, scope: !16)
!25 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!26 = !DILocalVariable(name: "b", scope: !7, file: !4, line: 1234, type: !25)

