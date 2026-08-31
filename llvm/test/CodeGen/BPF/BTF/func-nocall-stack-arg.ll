; RUN: llc -mtriple=bpfel -mcpu=v3 -filetype=obj -o %t1 %s
; RUN: llvm-objcopy --dump-section='.BTF'=%t2 %t1
; RUN: %python %p/print_btf.py %t2 | FileCheck -check-prefixes=CHECK %s

; Simulates DeadArgElimination on a function with 8 args (5 reg + 3 stack):
;   static __noinline int sub(int a1, int dead2, int a3, int a4,
;                             int a5, int a6, int dead7, int a8) {
;     return a1 + a3 + a4 + a5 + a6 + a8;
;   }
; DAE removes 'dead2' (register arg 2) and 'dead7' (stack arg 7).
; The remaining 6 args use R1-R5 + stack. BTF should emit FUNC_PROTO
; with (a1, a3, a4, a5, a6, a8).

; CHECK:      [1] INT 'int' size=4 bits_offset=0 nr_bits=32 encoding=SIGNED
; CHECK-NEXT: [2] FUNC_PROTO '(anon)' ret_type_id=1 vlen=6
; CHECK-NEXT: 	'a1' type_id=1
; CHECK-NEXT: 	'a3' type_id=1
; CHECK-NEXT: 	'a4' type_id=1
; CHECK-NEXT: 	'a5' type_id=1
; CHECK-NEXT: 	'a6' type_id=1
; CHECK-NEXT: 	'a8' type_id=1
; CHECK-NEXT: [3] FUNC 'sub' type_id=2 linkage=static

define internal i32 @sub(i32 %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5) #0 !dbg !7 {
  call void @llvm.dbg.value(metadata i32 %0, metadata !12, metadata !DIExpression()), !dbg !20
  call void @llvm.dbg.value(metadata i32 %1, metadata !14, metadata !DIExpression()), !dbg !20
  call void @llvm.dbg.value(metadata i32 %2, metadata !15, metadata !DIExpression()), !dbg !20
  call void @llvm.dbg.value(metadata i32 %3, metadata !16, metadata !DIExpression()), !dbg !20
  call void @llvm.dbg.value(metadata i32 %4, metadata !17, metadata !DIExpression()), !dbg !20
  call void @llvm.dbg.value(metadata i32 %5, metadata !19, metadata !DIExpression()), !dbg !20
  %7 = add i32 %0, %1, !dbg !21
  %8 = add i32 %7, %2, !dbg !21
  %9 = add i32 %8, %3, !dbg !21
  %10 = add i32 %9, %4, !dbg !21
  %11 = add i32 %10, %5, !dbg !21
  ret i32 %11, !dbg !22
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

attributes #0 = { noinline }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "t.c", directory: "/DNE")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang"}
!7 = distinct !DISubprogram(name: "sub", scope: !1, file: !1, line: 1, type: !8, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(cc: DW_CC_nocall, types: !9)
!9 = !{!10, !10, !10, !10, !10, !10, !10, !10, !10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{!12, !13, !14, !15, !16, !17, !18, !19}
!12 = !DILocalVariable(name: "a1", arg: 1, scope: !7, file: !1, line: 1, type: !10)
!13 = !DILocalVariable(name: "dead2", arg: 2, scope: !7, file: !1, line: 1, type: !10)
!14 = !DILocalVariable(name: "a3", arg: 3, scope: !7, file: !1, line: 1, type: !10)
!15 = !DILocalVariable(name: "a4", arg: 4, scope: !7, file: !1, line: 1, type: !10)
!16 = !DILocalVariable(name: "a5", arg: 5, scope: !7, file: !1, line: 1, type: !10)
!17 = !DILocalVariable(name: "a6", arg: 6, scope: !7, file: !1, line: 1, type: !10)
!18 = !DILocalVariable(name: "dead7", arg: 7, scope: !7, file: !1, line: 1, type: !10)
!19 = !DILocalVariable(name: "a8", arg: 8, scope: !7, file: !1, line: 1, type: !10)
!20 = !DILocation(line: 1, column: 1, scope: !7)
!21 = !DILocation(line: 2, column: 10, scope: !7)
!22 = !DILocation(line: 2, column: 3, scope: !7)
