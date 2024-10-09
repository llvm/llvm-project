; RUN: llc %s --filetype=obj -o %t
; RUN: llvm-objdump -d %t | FileCheck %s --check-prefixes=OBJ
; RUN: llvm-dwarfdump --debug-line %t | FileCheck %s --check-prefixes=DBG
; RUN: llc %s -o - | FileCheck %s --check-prefixes=ASM

; OBJ: 1:{{.*}}nop

;;     Address            Line   Column File   ISA Discriminator OpIndex Flags
; DBG: 0x0000000000000000      3      0      0   0             0       0  is_stmt
; DBG: 0x0000000000000001      0      0      0   0             0       0
; DBG: 0x0000000000000010      5      0      0   0             0       0  is_stmt prologue_end
; DBG: 0x0000000000000017      5      0      0   0             0       0  is_stmt end_sequence

; ASM:      .loc    0 0 0 is_stmt 0
; ASM-NEXT: .L{{.*}}:
; ASM-NEXT: .p2align        4

;; $ cat test.cpp
;; void g();
;; void f() {
;;   [[clang::code_align(16)]]
;;   while (1) { g(); }
;; }

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define dso_local void @f() local_unnamed_addr !dbg !9 {
entry:
  br label %while.body, !dbg !12

while.body:                                       ; preds = %entry, %while.body
  tail call void @g(), !dbg !12
  br label %while.body, !dbg !12, !llvm.loop !13
}

declare !dbg !16 void @g() local_unnamed_addr

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 19.0.0git", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{!"clang version 19.0.0git"}
!9 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 3, type: !10, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!10 = !DISubroutineType(types: !11)
!11 = !{}
!12 = !DILocation(line: 5, scope: !9)
!13 = distinct !{!13, !12, !12, !14, !15}
!14 = !{!"llvm.loop.mustprogress"}
!15 = !{!"llvm.loop.align", i32 16}
!16 = !DISubprogram(name: "g", scope: !1, file: !1, line: 2, type: !10, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
