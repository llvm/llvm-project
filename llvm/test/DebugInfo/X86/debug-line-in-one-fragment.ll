; RUN: llc %s -o %t.o -filetype=obj
; RUN: llvm-dwarfdump --debug-line %t.o | FileCheck %s --check-prefix=LINES
; RUN: llc %s -o %t.o -filetype=obj -debug-only=mc-dump 2>&1 | FileCheck %s --check-prefix=FRAGMENTS

;; Test (using mc-dump debug output) that .debug_line can be arranged in memory
;; using a single data fragment for a simple function, instead of using multiple
;; MCDwarfFragment fragments in un-necessary cirucmstances. Some targets want
;; multiple fragments so that they can linker-relax the linetable, but x86
;; doesn't.
;;
;; First, sanity check that the linetable output is as expected,

; LINES:      Address            Line   Column File   ISA Discriminator OpIndex Flags
; LINES-NEXT: ------------------ ------ ------ ------ --- ------------- ------- -------------
; LINES-NEXT: 0x0000000000000000      3      5      0   0             0       0  is_stmt prologue_end
; LINES-NEXT: 0x0000000000000003      4     12      0   0             0       0  is_stmt
; LINES-NEXT: 0x0000000000000007      4      3      0   0             0       0
; LINES-NEXT: 0x0000000000000008      4      3      0   0             0       0  end_sequence

;; Here's a typical example of .debug_line in a suboptimal arrangement: for each
;; address-delta there's an MCDwarfFragment computing the delta during
;; relaxation.
;;
;;    <MCSection Name:.debug_line Fragments:[
;;      <MCDataFragment<MCFragment
;;        Contents:[...
;;        Fixups:[...
;;      <MCDwarfFragment<MCFragment 0x5624fe435a80 LayoutOrder:1 Offset:86 HasInstructions:0 BundlePadding:0>
;;        AddrDelta:- LineDelta:1>,
;;      <MCDataFragment<MCFragment 0x5624fe435b00 LayoutOrder:2 Offset:87 HasInstructions:0 BundlePadding:0>
;;        Contents:[05,03,06] (3 bytes)>,
;;      <MCDwarfFragment<MCFragment 0x5624fe435bd0 LayoutOrder:3 Offset:90 HasInstructions:0 BundlePadding:0>
;;        AddrDelta:- LineDelta:0>,
;;      <MCDwarfFragment<MCFragment 0x5624fe435c50 LayoutOrder:4 Offset:91 HasInstructions:0 BundlePadding:0>
;;        AddrDelta:- LineDelta:9223372036854775807>,
;;      <MCDataFragment<MCFragment 0x5624fe435cd0 LayoutOrder:5 Offset:96 HasInstructions:0 BundlePadding:0>
;;        Contents:[] (0 bytes)>]>,
;;
;; The function in question is made of a single data fragment where the address
;; deltas are known at assembly time. We can (and should) emit .debug_line as a
;; single data fragment. (Check that we see one data fragment, then no more
;; fragments until the next section).
;
; FRAGMENTS:       <MCSection Name:.debug_line Fragments:[
; FRAGMENTS-NEXT:    <MCDataFragment<MCFragment
; FRAGMENTS-NEXT:      Contents:[
; FRAGMENTS-NEXT:      Fixups:[
;
; FRAGMENTS-NOT: MCDataFragment
; FRAGMENTS-NOT: MCFragment
;
; FRAGMENTS:     <MCSection Name:.debug_line_str Fragments:[

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define dso_local i32 @foo(i32 noundef %a, i32 noundef %b) local_unnamed_addr #0 !dbg !10 {
entry:
    #dbg_value(i32 %a, !15, !DIExpression(), !17)
    #dbg_value(i32 %b, !16, !DIExpression(), !17)
  %add = add nsw i32 %a, 1, !dbg !18
    #dbg_value(i32 %add, !15, !DIExpression(), !17)
  %mul = mul nsw i32 %b, 5, !dbg !19
    #dbg_value(i32 %mul, !16, !DIExpression(), !17)
  %add1 = add nsw i32 %add, %mul, !dbg !20
  ret i32 %add1, !dbg !21
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 21.0.0git (/fast/fs/llvm4 8bd196e0eaf4310ad2d9598512d13220b28b9aee)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "text.c", directory: "/fast/fs/llvm-main", checksumkind: CSK_MD5, checksum: "e81ed96ea393640bf1b965103b190e09")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!9 = !{!"clang version 21.0.0git (/fast/fs/llvm4 8bd196e0eaf4310ad2d9598512d13220b28b9aee)"}
!10 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !11, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !14)
!11 = !DISubroutineType(types: !12)
!12 = !{!13, !13, !13}
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !{!15, !16}
!15 = !DILocalVariable(name: "a", arg: 1, scope: !10, file: !1, line: 1, type: !13)
!16 = !DILocalVariable(name: "b", arg: 2, scope: !10, file: !1, line: 1, type: !13)
!17 = !DILocation(line: 0, scope: !10)
!18 = !DILocation(line: 2, column: 5, scope: !10)
!19 = !DILocation(line: 3, column: 5, scope: !10)
!20 = !DILocation(line: 4, column: 12, scope: !10)
!21 = !DILocation(line: 4, column: 3, scope: !10)
