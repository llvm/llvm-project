; RUN: llc %s -o - -stop-before=finalize-isel -march=x86-64 \
; RUN: | FileCheck %s --check-prefixes=INSTRREFON
; RUN: llc %s -o - -stop-before=finalize-isel -march=x86-64 \
; RUN:    -experimental-debug-variable-locations=true \
; RUN: | FileCheck %s --check-prefixes=INSTRREFON

; RUN: llc %s -o - -stop-before=finalize-isel -march=x86-64 \
; RUN:    -experimental-debug-variable-locations=false \
; RUN: | FileCheck %s --check-prefixes=INSTRREFOFF \
; RUN:    --implicit-check-not=DBG_INSTR_REF

;; This test checks that for an x86 triple, instruction referencing is used
;; by llc by default, and that it can be turned explicitly on or off as
;; desired.

; INSTRREFON: debugInstrRef: true
; INSTRREFON: DBG_INSTR_REF
; INSTRREFOFF: debugInstrRef: false
; INSTRREFOFF: DBG_VALUE

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

define hidden i32 @foo(i32 %a) local_unnamed_addr !dbg !7 {
  %b = add i32 %a, 1
  call void @llvm.dbg.value(metadata i32 %b, metadata !11, metadata !DIExpression()), !dbg !12
  ret i32 %b, !dbg !12
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "foo.cpp", directory: ".")
!2 = !DIBasicType(name: "int", size: 8, encoding: DW_ATE_signed)
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 2}
!6 = !{i32 7, !"PIC Level", i32 2}
!7 = distinct !DISubprogram(name: "foo", linkageName: "foo", scope: !1, file: !1, line: 6, type: !8, scopeLine: 6, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !10)
!8 = !DISubroutineType(types: !9)
!9 = !{!2, !2}
!10 = !{!11}
!11 = !DILocalVariable(name: "baz", scope: !7, file: !1, line: 7, type: !2)
!12 = !DILocation(line: 10, scope: !7)
