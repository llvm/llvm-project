; RUN: llc -debug -mtriple x86_64-apple-darwin < %s -o /dev/null 2>&1 | FileCheck %s
; REQUIRES: asserts
; Check that llc -debug actually prints variables and locations, rather than
; crashing.

; CHECK: DBG_VALUE

; Generated using `clang -g -O2 -S -emit-llvm -g` on the following source:
;
; static int foo(int x) { return x; }
; int bar(int x) { return foo(x); }

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

; Function Attrs: nounwind readnone ssp uwtable
define i32 @bar(i32 %x) #0 !dbg !4 {
entry:
  tail call void @llvm.dbg.value(metadata i32 %x, metadata !9, metadata !17), !dbg !18
  tail call void @llvm.dbg.value(metadata i32 %x, metadata !19, metadata !17), !dbg !21
  ret i32 %x, !dbg !22
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { nounwind readnone ssp uwtable "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="core2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!13, !14, !15}
!llvm.ident = !{!16}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.7.0 (trunk 233919) (llvm/trunk 233920)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "t.c", directory: "/Users/dexonsmith/data/llvm/debug-info/test/DebugInfo/X86")
!2 = !{}
!4 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 2, type: !5, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !8)
!5 = !DISubroutineType(types: !6)
!6 = !{!7, !7}
!7 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!8 = !{!9}
!9 = !DILocalVariable(name: "x", arg: 1, scope: !4, file: !1, line: 2, type: !7)
!10 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !5, isLocal: true, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !11)
!11 = !{!12}
!12 = !DILocalVariable(name: "x", arg: 1, scope: !10, file: !1, line: 1, type: !7)
!13 = !{i32 2, !"Dwarf Version", i32 2}
!14 = !{i32 2, !"Debug Info Version", i32 3}
!15 = !{i32 1, !"PIC Level", i32 2}
!16 = !{!"clang version 3.7.0 (trunk 233919) (llvm/trunk 233920)"}
!17 = !DIExpression()
!18 = !DILocation(line: 2, column: 13, scope: !4)
!19 = !DILocalVariable(name: "x", arg: 1, scope: !10, file: !1, line: 1, type: !7)
!20 = distinct !DILocation(line: 2, column: 25, scope: !4)
!21 = !DILocation(line: 1, column: 20, scope: !10, inlinedAt: !20)
!22 = !DILocation(line: 2, column: 18, scope: !4)
