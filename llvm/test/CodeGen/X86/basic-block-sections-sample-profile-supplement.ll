; RUN: echo 'v1' > %t.propeller
; RUN: echo 'f foo' >> %t.propeller
; RUN: echo 'g 0:100,1:0,2:100 1:0,2:0 2:100' >> %t.propeller
; RUN: echo 'foo:2000:1000' > %t.prof
; RUN: echo ' 3.1: 5' >> %t.prof
; RUN: echo 'foo:2000:1000' > %t.fs.prof
; RUN: echo ' 3.67108866: 5' >> %t.fs.prof
; RUN: echo 'foo:2000:1000' > %t.round_zero.prof
; RUN: echo ' 3.1: 4' >> %t.round_zero.prof
; RUN: echo 'foo:2000:1000' > %t.probe.prof
; RUN: echo ' 3.1: 5' >> %t.probe.prof
; RUN: echo ' !CFGChecksum: 123' >> %t.probe.prof

; Case 1: Without a sample profile supplement, block 1 has 0 Propeller samples and is split into cold section.
; RUN: llc < %s -O0 -mtriple=x86_64-pc-linux -function-sections -basic-block-sections=%t.propeller -basic-block-section-match-infer -propeller-match-bbid | \
; RUN: FileCheck %s --check-prefixes=CHECK,WITHOUT-SUPPLEMENT

; Case 2: With a sample profile supplement (using a nonzero base discriminator), block 1 gets its samples scaled to the Propeller entry weight (5 * 100 / 1000 rounds to 1), and is kept in the main section even without even flow distribution. When the scaled weight rounds to 0 (4 * 100 / 1000), block 1 stays in the cold section.
; RUN: llc < %s -O0 -mtriple=x86_64-pc-linux -function-sections -basic-block-sections=%t.propeller -basic-block-section-match-infer -propeller-match-bbid -sample-profile-supplement=%t.prof | \
; RUN: FileCheck %s --check-prefixes=CHECK,WITH-SUPPLEMENT
; RUN: llc < %s -O0 -mtriple=x86_64-pc-linux -function-sections -basic-block-sections=%t.propeller -basic-block-section-match-infer -propeller-match-bbid -sample-profile-even-flow-distribution=false -sample-profile-supplement=%t.prof | \
; RUN: FileCheck %s --check-prefixes=CHECK,WITH-SUPPLEMENT
; RUN: llc < %s -O0 -mtriple=x86_64-pc-linux -function-sections -basic-block-sections=%t.propeller -basic-block-section-match-infer -propeller-match-bbid -sample-profile-supplement=%t.round_zero.prof | \
; RUN: FileCheck %s --check-prefixes=CHECK,WITHOUT-SUPPLEMENT

; Case 3: A missing supplemental sample profile is diagnosed.
; RUN: not llc < %s -O0 -mtriple=x86_64-pc-linux -function-sections -basic-block-sections=%t.propeller -basic-block-section-match-infer -propeller-match-bbid -sample-profile-supplement=%t.missing 2>&1 | \
; RUN: FileCheck %s --check-prefix=MISSING
; MISSING: error: {{.*}}.missing: Could not open supplemental sample profile:

; Case 4: The supplemental sample profile is not read without -basic-block-section-match-infer or -propeller-match-bbid.
; RUN: llc < %s -O0 -mtriple=x86_64-pc-linux -function-sections -basic-block-sections=%t.propeller -propeller-match-bbid -sample-profile-supplement=%t.missing 2>&1 | \
; RUN: FileCheck %s --check-prefix=UNUSED
; RUN: llc < %s -O0 -mtriple=x86_64-pc-linux -function-sections -basic-block-sections=%t.propeller -basic-block-section-match-infer -sample-profile-supplement=%t.missing 2>&1 | \
; RUN: FileCheck %s --check-prefix=UNUSED
; UNUSED-NOT: Could not open supplemental sample profile

; Case 5: Supplemental sample profiles are ignored with a warning and not read when used with SampleFDO.
; The IR profile loader reads the primary profile in opt. Missing supplemental profile passed to llc is not read.
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%t.prof | \
; RUN: llc -O0 -mtriple=x86_64-pc-linux -function-sections -basic-block-sections=%t.propeller -basic-block-section-match-infer -propeller-match-bbid -sample-profile-supplement=%t.missing 2>&1 | \
; RUN: FileCheck %s --check-prefixes=SAMPLEFDO,CHECK,WITHOUT-SUPPLEMENT
; The MIR profile loader reads the FS primary profile in llc. Pseudo-probe-based supplemental profile is not read.
; RUN: llc < %s -O0 -mtriple=x86_64-pc-linux -function-sections -basic-block-sections=%t.propeller -basic-block-section-match-infer -propeller-match-bbid -enable-fs-discriminator -profile-isfs -fs-profile-file=%t.fs.prof -sample-profile-supplement=%t.probe.prof 2>&1 | \
; RUN: FileCheck %s --check-prefixes=SAMPLEFDO,CHECK,WITHOUT-SUPPLEMENT
; SAMPLEFDO: warning: {{.*}}: Supplemental sample profiles are not supported with SampleFDO

; Case 6: Flow-sensitive supplemental sample profiles and -enable-fs-discriminator must match.
; RUN: llc < %s -O0 -mtriple=x86_64-pc-linux -function-sections -basic-block-sections=%t.propeller -basic-block-section-match-infer -propeller-match-bbid -profile-isfs -sample-profile-supplement=%t.fs.prof 2>&1 | \
; RUN: FileCheck %s --check-prefixes=FS-WITHOUT-ENABLE,CHECK,WITHOUT-SUPPLEMENT
; RUN: llc < %s -O0 -mtriple=x86_64-pc-linux -function-sections -basic-block-sections=%t.propeller -basic-block-section-match-infer -propeller-match-bbid -enable-fs-discriminator -sample-profile-supplement=%t.prof 2>&1 | \
; RUN: FileCheck %s --check-prefixes=NON-FS-WITH-ENABLE,CHECK,WITHOUT-SUPPLEMENT
; RUN: llc < %s -O0 -mtriple=x86_64-pc-linux -function-sections -basic-block-sections=%t.propeller -basic-block-section-match-infer -propeller-match-bbid -profile-isfs -enable-fs-discriminator -sample-profile-supplement=%t.fs.prof | \
; RUN: FileCheck %s --check-prefixes=CHECK,WITH-SUPPLEMENT
; FS-WITHOUT-ENABLE: warning: {{.*}}.fs.prof: Flow-sensitive supplemental sample profiles require -enable-fs-discriminator
; NON-FS-WITH-ENABLE: warning: {{.*}}.prof: Non-flow-sensitive supplemental sample profiles are not supported with -enable-fs-discriminator

; Case 7: Context-sensitive and pseudo-probe supplemental sample profiles are ignored with a warning.
; RUN: echo '[foo]:2000:1000' > %t.cs.prof
; RUN: echo ' 3.1: 5' >> %t.cs.prof
; RUN: llc < %s -O0 -mtriple=x86_64-pc-linux -function-sections -basic-block-sections=%t.propeller -basic-block-section-match-infer -propeller-match-bbid -sample-profile-supplement=%t.cs.prof 2>&1 | \
; RUN: FileCheck %s --check-prefixes=CS-PROF,CHECK,WITHOUT-SUPPLEMENT
; CS-PROF: warning: {{.*}}.cs.prof: Context-sensitive supplemental sample profiles are not supported
; RUN: llc < %s -O0 -mtriple=x86_64-pc-linux -function-sections -basic-block-sections=%t.propeller -basic-block-section-match-infer -propeller-match-bbid -sample-profile-supplement=%t.probe.prof 2>&1 | \
; RUN: FileCheck %s --check-prefixes=PROBE-PROF,CHECK,WITHOUT-SUPPLEMENT
; PROBE-PROF: warning: {{.*}}.probe.prof: Pseudo probe based supplemental sample profiles are not supported

; Case 8: Functions with neither entry-block samples nor head samples in the supplement are skipped, while entry-block body samples suffice when head samples are 0.
; RUN: echo 'foo:2000:0' > %t.no_entry.prof
; RUN: echo ' 3.1: 5' >> %t.no_entry.prof
; RUN: llc < %s -O0 -mtriple=x86_64-pc-linux -function-sections -basic-block-sections=%t.propeller -basic-block-section-match-infer -propeller-match-bbid -sample-profile-supplement=%t.no_entry.prof | \
; RUN: FileCheck %s --check-prefixes=CHECK,WITHOUT-SUPPLEMENT
; RUN: echo 'foo:2000:0' > %t.entry_body.prof
; RUN: echo ' 0: 1000' >> %t.entry_body.prof
; RUN: echo ' 3.1: 5' >> %t.entry_body.prof
; RUN: llc < %s -O0 -mtriple=x86_64-pc-linux -function-sections -basic-block-sections=%t.propeller -basic-block-section-match-infer -propeller-match-bbid -sample-profile-supplement=%t.entry_body.prof | \
; RUN: FileCheck %s --check-prefixes=CHECK,WITH-SUPPLEMENT

define void @foo(i1 zeroext %0) nounwind !dbg !4 {
  %2 = alloca i8, align 1
  %3 = zext i1 %0 to i8
  store i8 %3, ptr %2, align 1, !dbg !10
  %4 = load i8, ptr %2, align 1, !dbg !11
  %5 = trunc i8 %4 to i1, !dbg !11
  br i1 %5, label %8, label %6, !dbg !12

6:                                                ; preds = %1
  %7 = call i32 @baz(), !dbg !13
  br label %8, !dbg !14

8:                                                ; preds = %6, %1
  ret void, !dbg !15
}

declare i32 @baz()

; CHECK:                    .section        .text.foo,"ax",@progbits
; CHECK-LABEL:              foo:
; CHECK-LABEL:              # %bb.0:
; CHECK-LABEL:              .LBB0_2:
; CHECK:                    retq

; WITHOUT-SUPPLEMENT:       .section        .text.split.foo,"ax",@progbits
; WITHOUT-SUPPLEMENT-LABEL: foo.cold:
; WITHOUT-SUPPLEMENT:       callq   baz@PLT

; WITH-SUPPLEMENT-LABEL:    .LBB0_1:
; WITH-SUPPLEMENT:          callq   baz@PLT
; WITH-SUPPLEMENT-NOT:      .section        .text.split.foo

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "test.cc", directory: "/tmp")
!2 = !{i32 2, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !5, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
; 67108866 = (1 << 26) | 2: non-FS prefix encoding decodes to base discriminator 1,
; while FS mode reads the full discriminator 67108866 (base 2 + PassLast bit 26).
!7 = distinct !DILexicalBlockFile(scope: !4, file: !1, discriminator: 67108866)
!10 = !DILocation(line: 1, column: 1, scope: !4)
!11 = !DILocation(line: 2, column: 1, scope: !4)
!12 = !DILocation(line: 3, column: 1, scope: !4)
!13 = !DILocation(line: 4, column: 1, scope: !7)
!14 = !DILocation(line: 5, column: 1, scope: !4)
!15 = !DILocation(line: 6, column: 1, scope: !4)
