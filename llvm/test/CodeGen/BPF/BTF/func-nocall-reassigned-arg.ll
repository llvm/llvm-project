; RUN: llc -mtriple=bpfel -mcpu=v3 -filetype=obj -o %t1 %s
; RUN: llvm-objcopy --dump-section='.BTF'=%t2 %t1
; RUN: %python %p/print_btf.py %t2 | FileCheck -check-prefixes=CHECK %s

; DeadArgElimination on a function that reassigns one argument to another:
;   static __noinline int sub(int unused, int a, int b) {
;     int t = a * 3;
;     a = b;
;     return t + a;
;   }
; 'a' gets a second DBG_VALUE for the reassignment, naming b's register R2.
; Only the first one is its entry location, so 'a' must stay bound to R1.

; CHECK:      [1] INT 'int' size=4 bits_offset=0 nr_bits=32 encoding=SIGNED
; CHECK-NEXT: [2] FUNC_PROTO '(anon)' ret_type_id=1 vlen=2
; CHECK-NEXT: 	'a' type_id=1
; CHECK-NEXT: 	'b' type_id=1
; CHECK-NEXT: [3] FUNC 'sub' type_id=2 linkage=static

define internal i32 @sub(i32 %0, i32 %1) #0 !dbg !7 {
    #dbg_value(i32 %0, !13, !DIExpression(), !15)
    #dbg_value(i32 %1, !14, !DIExpression(), !15)
  %3 = mul i32 %0, 3, !dbg !16
    #dbg_value(i32 %1, !13, !DIExpression(), !15)
  %4 = add i32 %3, %1, !dbg !16
  ret i32 %4, !dbg !17
}

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
!9 = !{!10, !10, !10, !10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{!12, !13, !14}
!12 = !DILocalVariable(name: "unused", arg: 1, scope: !7, file: !1, line: 1, type: !10)
!13 = !DILocalVariable(name: "a", arg: 2, scope: !7, file: !1, line: 1, type: !10)
!14 = !DILocalVariable(name: "b", arg: 3, scope: !7, file: !1, line: 1, type: !10)
!15 = !DILocation(line: 1, column: 1, scope: !7)
!16 = !DILocation(line: 2, column: 10, scope: !7)
!17 = !DILocation(line: 4, column: 3, scope: !7)
