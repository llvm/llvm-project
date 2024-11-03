; RUN: llc -mtriple=x86_64-unknown-unknown -start-after=codegenprepare -stop-before=finalize-isel %s -o - -experimental-debug-variable-locations=false | FileCheck %s --check-prefixes=COMMON,CHECK
; RUN: llc -mtriple=x86_64-unknown-unknown -start-after=codegenprepare -stop-before=finalize-isel %s -o - -experimental-debug-variable-locations=true | FileCheck %s --check-prefixes=COMMON,INSTRREF

; Test the movement of dbg.values of arguments. SelectionDAG tries to be
; helpful and places DBG_VALUEs of Arguments at the start of functions.
; Unfortunately, this doesn't necessarily make sense, as one can specify an
; Argument IR Value as a variable location anywhere in the program.
;
; Distinguish cases where we want to hoist DBG_VALUEs, and those where we
; don't, by whether the referred to variable is a parameter to the current
; function. In the test below, 'xyzzy' is a parameter to an inlined function,
; but should not be hoisted to the start of the function.
;
; With instruction referencing, accuracy becomes more important than coverage,
; so debug instructions are placed wherever they were in the IR.
;
; Original test case, in which 'xyzzy' became unavailable because its DBG_VALUE
; landed far from any uses, compiled "clang -O2 -g" with inlining,
;
;    int ext(void);
;
;    int
;    foo(int xyzzy)
;    {
;      xyzzy = ext() * xyzzy;
;      xyzzy += 1;
;      ext();
;      return xyzzy;
;    }
;
;    int
;    bar(int baz, int qux)
;    {
;      int fish;
;      switch (qux) {
;      case 12:        fish = 8;        break;
;      case 848:       fish = 0;        break;
;      case 99999:     fish = -1;       break;
;      default:        fish = 12;
;      }
;      qux %= fish;
;      qux += foo(baz);
;      return qux;
;    }

; COMMON: [[BAZVAR:![0-9]+]] = !DILocalVariable(name: "baz",
; COMMON: [[XYZVAR:![0-9]+]] = !DILocalVariable(name: "xyzzy",

; Start of MIR function block,
; CHECK-LABEL: body
; Expect DBG_VALUE of physreg,
; CHECK:       DBG_VALUE $edi, $noreg, [[BAZVAR]]
; Expect DBG_VALUE of virtreg,
; CHECK:       DBG_VALUE [[ARGREG:%[0-9]+]], $noreg, [[BAZVAR]]
; Label for next block,
; CHECK-LABEL: bb.1.next
; Correctly place dbg.value in the 'next' block.
; CHECK:       DBG_VALUE [[ARGREG]], $noreg, [[XYZVAR]]

; INSTRREF-LABEL: body
; INSTRREF: DBG_PHI $edi, 1
; INSTRREF: DBG_VALUE $edi, $noreg, [[BAZVAR]]
; INSTRREF-LABEL: bb.1.next
; INSTRREF: DBG_INSTR_REF [[XYZVAR]], {{.+}}, dbg-instr-ref(1, 0)

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare i32 @ext()

define dso_local i32 @bar(i32, i32) local_unnamed_addr !dbg !7 {
  %3 = srem i32 %1, %0, !dbg !15
  call void @llvm.dbg.value(metadata i32 %0, metadata !12, metadata !DIExpression()), !dbg !16
  br label %next

next:                                             ; preds = %2
  call void @llvm.dbg.value(metadata i32 %0, metadata !17, metadata !DIExpression()), !dbg !22
  %4 = tail call i32 @ext(), !dbg !24
  %5 = mul nsw i32 %3, %4, !dbg !25
  %6 = add i32 %5, %0, !dbg !25
  ret i32 %6, !dbg !25
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, globals: !2, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: ".")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 8.0.0"}
!7 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 19, type: !8, scopeLine: 20, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !10, !10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{!12}
!12 = !DILocalVariable(name: "baz", arg: 1, scope: !7, file: !1, line: 19, type: !10)
!15 = !DILocation(line: 35, column: 7, scope: !7)
!16 = !DILocation(line: 19, column: 9, scope: !7)
!17 = !DILocalVariable(name: "xyzzy", arg: 1, scope: !18, file: !1, line: 10, type: !10)
!18 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 10, type: !19, scopeLine: 11, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !21)
!19 = !DISubroutineType(types: !20)
!20 = !{!10, !10}
!21 = !{!17}
!22 = !DILocation(line: 10, column: 9, scope: !18, inlinedAt: !23)
!23 = distinct !DILocation(line: 36, column: 10, scope: !7)
!24 = !DILocation(line: 12, column: 11, scope: !18, inlinedAt: !23)
!25 = !DILocation(line: 37, column: 3, scope: !7)
