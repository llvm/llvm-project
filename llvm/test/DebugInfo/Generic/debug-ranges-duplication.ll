; AIX doesn't currently support DWARF 5 section .debug_rnglists
; XFAIL: target={{.*}}-zos{{.*}}, target={{.*}}-aix{{.*}}

; RUN: %llc_dwarf -O0 -filetype=obj < %s | llvm-dwarfdump -debug-info - | FileCheck %s
;
; Generated from the following C++ source with:
; clang -S -emit-llvm -g -O2 test.c
;
; /* BEGIN SOURCE */
; void f1();
; inline void f2() {
;   f1();
;   f1();
; }
; inline void f3() {
;   f2();
; }
; void f4() {
;   f3();
;   f1();
; }
; /* END SOURCE */
;
; Minor complication: after generating the LLVM IR, it was manually edited so
; that the 'f1()' call from f3 was reordered to appear between the two inlined
; f1 calls from f2. This causes f2's inlined_subroutine to use DW_AT_ranges.

; Check that identical debug ranges in succession reuse the same entry in
; .debug_ranges rather than emitting duplicate entries.

; CHECK:      DW_TAG_inlined_subroutine
; CHECK:      DW_AT_ranges
; CHECK-SAME: rangelist = 0x[[#%.8X,RANGE:]]
; CHECK:      DW_TAG_inlined_subroutine
; CHECK:      DW_AT_ranges
; CHECK-SAME: rangelist = 0x[[#RANGE]]

; Function Attrs: nounwind uwtable
define dso_local void @f4() local_unnamed_addr !dbg !9 {
entry:
  tail call void (...) @f1(), !dbg !12
  tail call void (...) @f1(), !dbg !18
  tail call void (...) @f1(), !dbg !17
  ret void, !dbg !19
}

declare !dbg !20 void @f1(...) local_unnamed_addr

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 20.0.0git (https://github.com/llvm/llvm-project.git 9edd998e10fabfff067b9e6e5b044f85a24d0dd5)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/home/khuey/dev/llvm-project", checksumkind: CSK_MD5, checksum: "4510feb241cf078af753e3dc13205127")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{!"clang version 20.0.0git (https://github.com/llvm/llvm-project.git 9edd998e10fabfff067b9e6e5b044f85a24d0dd5)"}
!9 = distinct !DISubprogram(name: "f4", scope: !1, file: !1, line: 9, type: !10, scopeLine: 9, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!10 = !DISubroutineType(types: !11)
!11 = !{null}
!12 = !DILocation(line: 3, column: 3, scope: !13, inlinedAt: !14)
!13 = distinct !DISubprogram(name: "f2", scope: !1, file: !1, line: 2, type: !10, scopeLine: 2, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!14 = distinct !DILocation(line: 7, column: 3, scope: !15, inlinedAt: !16)
!15 = distinct !DISubprogram(name: "f3", scope: !1, file: !1, line: 6, type: !10, scopeLine: 6, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!16 = distinct !DILocation(line: 10, column: 3, scope: !9)
!17 = !DILocation(line: 4, column: 3, scope: !13, inlinedAt: !14)
!18 = !DILocation(line: 11, column: 3, scope: !9)
!19 = !DILocation(line: 12, column: 1, scope: !9)
!20 = !DISubprogram(name: "f1", scope: !1, file: !1, line: 1, type: !10, spFlags: DISPFlagOptimized)
