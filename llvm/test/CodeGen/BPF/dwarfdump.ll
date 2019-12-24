; RUN: llc -O2 -march=bpfel %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-line %t | FileCheck %s
; RUN: llc -O2 -march=bpfeb %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-line %t | FileCheck %s

source_filename = "testprog.c"
target datalayout = "e-m:e-p:64:64-i64:64-n32:64-S128"
target triple = "bpf"

@testprog.myvar_c = internal unnamed_addr global i32 0, align 4, !dbg !0

; Function Attrs: nounwind
define i32 @testprog(i32, i32) local_unnamed_addr #0 !dbg !2 {
  tail call void @llvm.dbg.value(metadata i32 %0, i64 0, metadata !11, metadata !16), !dbg !17
  tail call void @llvm.dbg.value(metadata i32 %1, i64 0, metadata !12, metadata !16), !dbg !18
  %3 = load i32, i32* @testprog.myvar_c, align 4, !dbg !19, !tbaa !20
  %4 = add i32 %1, %0, !dbg !24
  %5 = add i32 %4, %3, !dbg !25
  store i32 %5, i32* @testprog.myvar_c, align 4, !dbg !26, !tbaa !20
  ret i32 %5, !dbg !27
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!7}
!llvm.module.flags = !{!13, !14}
!llvm.ident = !{!15}

!0 = distinct !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "myvar_c", scope: !2, file: !3, line: 3, type: !6, isLocal: true, isDefinition: true)
!2 = distinct !DISubprogram(name: "testprog", scope: !3, file: !3, line: 1, type: !4, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: true, unit: !7, retainedNodes: !10)
!3 = !DIFile(filename: "testprog.c", directory: "/w/llvm/bld")
!4 = !DISubroutineType(types: !5)
!5 = !{!6, !6, !6}
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 4.0.0 (trunk 287518) (llvm/trunk 287520)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !8, globals: !9)
!8 = !{}
!9 = !{!0}
!10 = !{!11, !12}
!11 = !DILocalVariable(name: "myvar_a", arg: 1, scope: !2, file: !3, line: 1, type: !6)
!12 = !DILocalVariable(name: "myvar_b", arg: 2, scope: !2, file: !3, line: 1, type: !6)
!13 = !{i32 2, !"Dwarf Version", i32 4}
!14 = !{i32 2, !"Debug Info Version", i32 3}
!15 = !{!"clang version 4.0.0 (trunk 287518) (llvm/trunk 287520)"}
!16 = !DIExpression()
!17 = !DILocation(line: 1, column: 18, scope: !2)
!18 = !DILocation(line: 1, column: 31, scope: !2)
!19 = !DILocation(line: 5, column: 19, scope: !2)
!20 = !{!21, !21, i64 0}
!21 = !{!"int", !22, i64 0}
!22 = !{!"omnipotent char", !23, i64 0}
!23 = !{!"Simple C/C++ TBAA"}
!24 = !DILocation(line: 5, column: 27, scope: !2)
!25 = !DILocation(line: 7, column: 27, scope: !2)
!26 = !DILocation(line: 7, column: 17, scope: !2)
!27 = !DILocation(line: 9, column: 9, scope: !2)

; CHECK: file_names[  1]:
; CHECK-NEXT: name: "testprog.c"
; CHECK-NEXT: dir_index: 0
; CHECK: 0x0000000000000000      2
; CHECK: 0x0000000000000028      7
