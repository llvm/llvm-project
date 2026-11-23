; REQUIRES: asserts && x86-registered-target
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%S/Inputs/pseudo-probe-stale-profile-backward-matching.prof --salvage-stale-profile -S --debug-only=sample-profile-impl,sample-profile-matcher 2>&1 | FileCheck %s

; Test that backward matching in matchNonCallsiteLocs correctly overwrites
; forward-matched entries for the second half of non-anchor locations between
; two matched anchors, using pseudo-probe-based profile.
;
; IR probes (ordered by probe ID):
;   1(block), 2(block), 3(call foo), 4(block), 5(block), 6(call bar), 7(block), 8(block)
;
; Profile has foo at probe 5 and bar at probe 12 (checksum mismatch triggers
; stale profile matching).
;
; Anchor matching by LCS:
;   foo: IR probe 3 -> Profile probe 5 (delta = +2)
;   bar: IR probe 6 -> Profile probe 12 (delta = +6)
;
; Block probes [4, 5] are non-anchors between foo and bar.
; Split evenly:
;   First half  [probe 4]: forward  delta=+2 -> maps to profile probe 6 (no samples)
;   Second half [probe 5]: backward delta=+6 -> maps to profile probe 11 (body sample 999)
;
; Without the fix, probe 5 stays at forward value 5+2=7 (body sample 111).
; With the fix (insert_or_assign), backward overwrites to 5+6=11 (body sample
; 999).

; Verify anchor and non-anchor matching:
; CHECK: Callsite with callee:foo is matched from 3 to 5
; CHECK: Location is matched from 4 to 6
; CHECK: Location is matched from 5 to 7
; CHECK: Callsite with callee:bar is matched from 6 to 12
; CHECK: Location is rematched backwards from 5 to 11

; Verify the sample weight at probe 5 uses backward-matched value (999).
; Without the fix, this would show 111.
; CHECK: call void @llvm.pseudoprobe(i64 6355742111584357505, i64 5, i32 0, i64 -1){{.*}}weight: 999

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare i32 @foo(i32)
declare i32 @bar(i32)

define dso_local i32 @test_backward_matching_probe() #0 !dbg !10 {
entry:
  call void @llvm.pseudoprobe(i64 6355742111584357505, i64 1, i32 0, i64 -1), !dbg !14
  call void @llvm.pseudoprobe(i64 6355742111584357505, i64 2, i32 0, i64 -1), !dbg !14
  ; call foo — probe ID 3 encoded in discriminator 186646559
  %r1 = call i32 @foo(i32 1), !dbg !15
  call void @llvm.pseudoprobe(i64 6355742111584357505, i64 4, i32 0, i64 -1), !dbg !14
  call void @llvm.pseudoprobe(i64 6355742111584357505, i64 5, i32 0, i64 -1), !dbg !14
  ; call bar — probe ID 6 encoded in discriminator 186646583
  %r2 = call i32 @bar(i32 2), !dbg !17
  call void @llvm.pseudoprobe(i64 6355742111584357505, i64 7, i32 0, i64 -1), !dbg !14
  call void @llvm.pseudoprobe(i64 6355742111584357505, i64 8, i32 0, i64 -1), !dbg !14
  %sum = add i32 %r1, %r2
  ret i32 %sum, !dbg !14
}

declare void @llvm.pseudoprobe(i64, i64, i32, i64) #1

attributes #0 = { nounwind uwtable "use-sample-profile" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6, !7}
!llvm.ident = !{!8}
!llvm.pseudo_probe_desc = !{!13}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 19.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test_probe_bm.c", directory: "/tmp")
!2 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!3 = !{i32 7, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 7, !"uwtable", i32 2}
!7 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!8 = !{!"clang version 19.0.0"}
!9 = !DIFile(filename: "test_probe_bm.c", directory: "/tmp")
!10 = distinct !DISubprogram(name: "test_backward_matching_probe", scope: !9, file: !9, line: 10, type: !11, scopeLine: 10, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!11 = !DISubroutineType(types: !12)
!12 = !{}
; GUID=6355742111584357505, CFGChecksum=123456 (mismatches profile's 999999)
!13 = !{i64 6355742111584357505, i64 123456, !"test_backward_matching_probe"}
; Generic debug location for block probes
!14 = !DILocation(line: 11, column: 3, scope: !10)
; call foo — discriminator 186646559 encodes probe ID 3, type=direct call
!15 = !DILocation(line: 12, column: 8, scope: !16)
!16 = !DILexicalBlockFile(scope: !10, file: !9, discriminator: 186646559)
; call bar — discriminator 186646583 encodes probe ID 6, type=direct call
!17 = !DILocation(line: 14, column: 8, scope: !18)
!18 = !DILexicalBlockFile(scope: !10, file: !9, discriminator: 186646583)
