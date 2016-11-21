; RUN: llc -O2 -march=bpfel %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-dump=line %t | FileCheck %s

source_filename = "testprog.c"
target datalayout = "e-m:e-p:64:64-i64:64-n32:64-S128"
target triple = "bpf"

@testprog.myvar_c = internal unnamed_addr global i32 0, align 4, !dbg !0

define i32 @testprog(i32, i32) local_unnamed_addr #0 !dbg !1 {
  tail call void @llvm.dbg.value(metadata i32 %0, i64 0, metadata !10, metadata !15), !dbg !16
  tail call void @llvm.dbg.value(metadata i32 %1, i64 0, metadata !11, metadata !15), !dbg !17
  %3 = load i32, i32* @testprog.myvar_c, align 4, !dbg !18, !tbaa !19
  %4 = add i32 %1, %0, !dbg !23
  %5 = add i32 %4, %3, !dbg !24
  store i32 %5, i32* @testprog.myvar_c, align 4, !dbg !25, !tbaa !19
  ret i32 %5, !dbg !26
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!6}
!llvm.module.flags = !{!12, !13}
!llvm.ident = !{!14}

!0 = distinct !DIGlobalVariable(name: "myvar_c", scope: !1, file: !2, line: 3, type: !5, isLocal: true, isDefinition: true)
!1 = distinct !DISubprogram(name: "testprog", scope: !2, file: !2, line: 1, type: !3, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: true, unit: !6, variables: !9)
!2 = !DIFile(filename: "testprog.c", directory: "/w/llvm/bld")
!3 = !DISubroutineType(types: !4)
!4 = !{!5, !5, !5}
!5 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!6 = distinct !DICompileUnit(language: DW_LANG_C99, file: !2, producer: "clang version 4.0.0 (trunk 287518) (llvm/trunk 287520)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !7, globals: !8)
!7 = !{}
!8 = !{!0}
!9 = !{!10, !11}
!10 = !DILocalVariable(name: "myvar_a", arg: 1, scope: !1, file: !2, line: 1, type: !5)
!11 = !DILocalVariable(name: "myvar_b", arg: 2, scope: !1, file: !2, line: 1, type: !5)
!12 = !{i32 2, !"Dwarf Version", i32 4}
!13 = !{i32 2, !"Debug Info Version", i32 3}
!14 = !{!"clang version 4.0.0 (trunk 287518) (llvm/trunk 287520)"}
!15 = !DIExpression()
!16 = !DILocation(line: 1, column: 18, scope: !1)
!17 = !DILocation(line: 1, column: 31, scope: !1)
!18 = !DILocation(line: 5, column: 19, scope: !1)
!19 = !{!20, !20, i64 0}
!20 = !{!"int", !21, i64 0}
!21 = !{!"omnipotent char", !22, i64 0}
!22 = !{!"Simple C/C++ TBAA"}
!23 = !DILocation(line: 5, column: 27, scope: !1)
!24 = !DILocation(line: 7, column: 27, scope: !1)
!25 = !DILocation(line: 7, column: 17, scope: !1)
!26 = !DILocation(line: 9, column: 9, scope: !1)
; CHECK: file_names[  1]    0 0x00000000 0x00000000 testprog.c
; CHECK: 0x0000000000000000      2
; CHECK: 0x0000000000000020      7
