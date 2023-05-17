; RUN: llc -start-after=codegenprepare -stop-before finalize-isel -o - %s \
; RUN:     -experimental-debug-variable-locations=false \
; RUN: | FileCheck %s --check-prefixes=CHECK,DBGVALUE
; RUN: llc -start-after=codegenprepare -stop-before finalize-isel -o - %s \
; RUN:     -experimental-debug-variable-locations=true \
; RUN: | FileCheck %s --check-prefixes=CHECK,INSTRREF

; This tests that transferDbgValues() changes order of SDDbgValue transferred
; to another node and debug info for 'ADD32ri' appears *after* the instruction.
;
; This test case was generated from the following program
; using: clang -g -O3 -S -emit-llvm test.c
;
; int foo(int a, int *b) {
;   int c = a + 512;
;   if (c != 0)
;     *b = a;
;   return c;
; }

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: bb.0.entry:
; CHECK:         %[[REG:[0-9]+]]:gr32 = ADD32ri %1, 512,
; INSTRREF-SAME:   debug-instr-number 1
; INSTRREF-NEXT: DBG_INSTR_REF {{.+}}, dbg-instr-ref(1, 0)
; DBGVALUE-NEXT: DBG_VALUE %[[REG]]

; Function Attrs: nofree norecurse nounwind uwtable writeonly
define dso_local i32 @foo(i32 %a, ptr nocapture %b) local_unnamed_addr !dbg !7 {
entry:
  %add = add nsw i32 %a, 512, !dbg !18
  call void @llvm.dbg.value(metadata i32 %add, metadata !16, metadata !DIExpression()), !dbg !17
  %cmp = icmp eq i32 %add, 0, !dbg !18
  br i1 %cmp, label %if.end, label %if.then, !dbg !18

if.then:                                          ; preds = %entry
  store i32 %a, ptr %b, align 4, !dbg !18
  br label %if.end, !dbg !18

if.end:                                           ; preds = %entry, %if.then
  ret i32 %add, !dbg !18
}

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 10.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 10.0.0"}
!7 = distinct !DISubprogram(name: "foo", scope: !8, file: !8, line: 1, type: !9, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !13)
!8 = !DIFile(filename: "test.c", directory: "/")
!9 = !DISubroutineType(types: !10)
!10 = !{!11, !11, !12}
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !11, size: 64)
!13 = !{!14, !15, !16}
!14 = !DILocalVariable(name: "a", arg: 1, scope: !7, file: !8, line: 1, type: !11)
!15 = !DILocalVariable(name: "b", arg: 2, scope: !7, file: !8, line: 1, type: !12)
!16 = !DILocalVariable(name: "c", scope: !7, file: !8, line: 2, type: !11)
!17 = !DILocation(line: 0, scope: !7)
!18 = !DILocation(line: 2, column: 13, scope: !7)
