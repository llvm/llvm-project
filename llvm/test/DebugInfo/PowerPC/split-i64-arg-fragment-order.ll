; RUN: llc -O1 -mtriple=powerpc-unknown-linux-gnu -stop-after=finalize-isel %s -o - | FileCheck %s
; RUN: llc -O1 -mtriple=powerpcle-unknown-linux-gnu -stop-after=finalize-isel %s -o - | FileCheck %s

; An i64 argument split across two registers is described by two
; fragments, and a fragment's offset counts from the variable's
; lowest-addressed byte. The PPC32 ABI passes the half at the lower
; address in r3 on both byte orders: the high word on big-endian, the
; low word on little-endian. So r3 must be fragment 0 and r4 fragment 32
; for both triples.
; getUnderlyingArgRegs used to walk BUILD_PAIR's operands in (low, high)
; order regardless of byte order and swapped the two halves on big-endian
; targets, so a debugger showed `x` with its words exchanged. MIPS, SPARC
; and ARMEB had the same defect; little-endian targets are unaffected.

; CHECK-DAG: [[R3:%[0-9]+]]:gprc = COPY $r3
; CHECK-DAG: [[R4:%[0-9]+]]:gprc = COPY $r4
; CHECK-DAG: DBG_VALUE [[R3]], $noreg, ![[X:[0-9]+]], !DIExpression(DW_OP_LLVM_fragment, 0, 32)
; CHECK-DAG: DBG_VALUE [[R4]], $noreg, ![[X]], !DIExpression(DW_OP_LLVM_fragment, 32, 32)

define i64 @f(i64 %x) !dbg !9 {
entry:
    #dbg_value(i64 %x, !14, !DIExpression(), !15)
  %call = tail call i64 @g(i64 %x), !dbg !16
  %add = add nsw i64 %x, 1, !dbg !16
  %call1 = tail call i64 @g(i64 %add), !dbg !16
  %add2 = add nsw i64 %call1, %call, !dbg !16
  ret i64 %add2, !dbg !16
}

declare i64 @g(i64)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "s.c", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!9 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 2, type: !10, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !13)
!10 = !DISubroutineType(types: !11)
!11 = !{!12, !12}
!12 = !DIBasicType(name: "long long", size: 64, encoding: DW_ATE_signed)
!13 = !{!14}
!14 = !DILocalVariable(name: "x", arg: 1, scope: !9, file: !1, line: 2, type: !12)
!15 = !DILocation(line: 0, scope: !9)
!16 = !DILocation(line: 2, scope: !9)
