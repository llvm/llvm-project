; RUN: llc -mtriple=bpfel -mcpu=v3 -filetype=obj -o %t1 %s
; RUN: llvm-objcopy --dump-section='.BTF'=%t2 %t1
; RUN: %python %p/print_btf.py %t2 | FileCheck -check-prefixes=CHECK %s

; Like func-nocall-stack-arg.ll but with i64 arguments so that
; stack-arg loads (LDD $r11, offset) and their DBG_VALUEs both use
; the 64-bit register ($r1), exercising the StackLoadRegs path in
; collectNocallEntryArgRegs rather than falling through on a
; sub-register mismatch.
;
; Source (after DAE removes dead2 and dead7):
;   static __noinline long sub(long a1, long dead2, long a3, long a4,
;                              long a5, long a6, long dead7, long a8) {
;     return a1 + a3 + a4 + a5 + a6 + a8;
;   }

; CHECK:      [1] INT 'long' size=8 bits_offset=0 nr_bits=64 encoding=SIGNED
; CHECK-NEXT: [2] FUNC_PROTO '(anon)' ret_type_id=1 vlen=6
; CHECK-NEXT: 	'a1' type_id=1
; CHECK-NEXT: 	'a3' type_id=1
; CHECK-NEXT: 	'a4' type_id=1
; CHECK-NEXT: 	'a5' type_id=1
; CHECK-NEXT: 	'a6' type_id=1
; CHECK-NEXT: 	'a8' type_id=1
; CHECK-NEXT: [3] FUNC 'sub' type_id=2 linkage=static

define internal i64 @sub(i64 %0, i64 %1, i64 %2, i64 %3, i64 %4, i64 %5) #0 !dbg !7 {
  call void @llvm.dbg.value(metadata i64 %0, metadata !12, metadata !DIExpression()), !dbg !20
  call void @llvm.dbg.value(metadata i64 %1, metadata !14, metadata !DIExpression()), !dbg !20
  call void @llvm.dbg.value(metadata i64 %2, metadata !15, metadata !DIExpression()), !dbg !20
  call void @llvm.dbg.value(metadata i64 %3, metadata !16, metadata !DIExpression()), !dbg !20
  call void @llvm.dbg.value(metadata i64 %4, metadata !17, metadata !DIExpression()), !dbg !20
  call void @llvm.dbg.value(metadata i64 %5, metadata !19, metadata !DIExpression()), !dbg !20
  %7 = add i64 %0, %1, !dbg !21
  %8 = add i64 %7, %2, !dbg !21
  %9 = add i64 %8, %3, !dbg !21
  %10 = add i64 %9, %4, !dbg !21
  %11 = add i64 %10, %5, !dbg !21
  ret i64 %11, !dbg !22
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
!10 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
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
