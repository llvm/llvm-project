; RUN: llc -mtriple=bpfel -mcpu=v3 -filetype=obj -o %t1 %s
; RUN: llvm-objcopy --dump-section='.BTF'=%t2 %t1
; RUN: %python %p/print_btf.py %t2 | FileCheck -check-prefixes=CHECK %s

; DeadArgElimination on a function whose live arguments outnumber the
; callee-saved registers, so one of them is spilled:
;   static __noinline int sub(int unused, _Bool b, int a1, int a2,
;                             int a3, int a4) {
;     sink(a1); sink(a2); sink(a3); sink(a4);
;     return b ? a1 : a2;
;   }
; 'b' arrives in R1 and is spilled, and the register allocator emits a second
; DBG_VALUE for the spill slot:
;   DBG_VALUE $w1, $noreg, !"b", !DIExpression(DW_OP_LLVM_convert, ...)
;   STW32 $w1, $r10, -8
;   DBG_VALUE $r10, $noreg, !"b", !DIExpression(DW_OP_constu, 8, DW_OP_minus,
;                                               DW_OP_deref_size, 4, ...)
; The second one is neither indirect nor uses a redefined register, so only
; taking the first DBG_VALUE keeps 'b' bound to R1.

; CHECK:      [1] INT 'int' size=4 bits_offset=0 nr_bits=32 encoding=SIGNED
; CHECK-NEXT: [2] INT '_Bool' size=1 bits_offset=0 nr_bits=8 encoding=BOOL
; CHECK-NEXT: [3] FUNC_PROTO '(anon)' ret_type_id=1 vlen=5
; CHECK-NEXT: 	'b' type_id=2
; CHECK-NEXT: 	'a1' type_id=1
; CHECK-NEXT: 	'a2' type_id=1
; CHECK-NEXT: 	'a3' type_id=1
; CHECK-NEXT: 	'a4' type_id=1
; CHECK-NEXT: [4] FUNC 'sub' type_id=3 linkage=static

define internal i32 @sub(i1 zeroext %0, i32 %1, i32 %2, i32 %3, i32 %4) #0 !dbg !7 {
    #dbg_value(i1 %0, !13, !DIExpression(DW_OP_LLVM_convert, 1, DW_ATE_unsigned, DW_OP_LLVM_convert, 8, DW_ATE_unsigned, DW_OP_stack_value), !19)
    #dbg_value(i32 %1, !14, !DIExpression(), !19)
    #dbg_value(i32 %2, !15, !DIExpression(), !19)
    #dbg_value(i32 %3, !16, !DIExpression(), !19)
    #dbg_value(i32 %4, !17, !DIExpression(), !19)
  call void @sink(i32 %1), !dbg !20
  call void @sink(i32 %2), !dbg !20
  call void @sink(i32 %3), !dbg !20
  call void @sink(i32 %4), !dbg !20
  %6 = select i1 %0, i32 %1, i32 %2, !dbg !20
  ret i32 %6, !dbg !21
}

declare void @sink(i32)

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
!9 = !{!10, !10, !18, !10, !10, !10, !10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{!12, !13, !14, !15, !16, !17}
!12 = !DILocalVariable(name: "unused", arg: 1, scope: !7, file: !1, line: 1, type: !10)
!13 = !DILocalVariable(name: "b", arg: 2, scope: !7, file: !1, line: 1, type: !18)
!14 = !DILocalVariable(name: "a1", arg: 3, scope: !7, file: !1, line: 1, type: !10)
!15 = !DILocalVariable(name: "a2", arg: 4, scope: !7, file: !1, line: 1, type: !10)
!16 = !DILocalVariable(name: "a3", arg: 5, scope: !7, file: !1, line: 1, type: !10)
!17 = !DILocalVariable(name: "a4", arg: 6, scope: !7, file: !1, line: 1, type: !10)
!18 = !DIBasicType(name: "_Bool", size: 8, encoding: DW_ATE_boolean)
!19 = !DILocation(line: 1, column: 1, scope: !7)
!20 = !DILocation(line: 3, column: 3, scope: !7)
!21 = !DILocation(line: 4, column: 3, scope: !7)
