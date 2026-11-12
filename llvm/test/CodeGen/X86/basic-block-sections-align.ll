;; Tests basic block alignment deferred to the basic block sections pass
;; (-align-blocks-in-bbsections).
;;
;; 1. Without -align-blocks-in-bbsections, MachineBlockPlacement assigns
;;    alignments before basic block sections based on static branch
;;    probabilities (since test_propeller has no IR PGO metadata), leaving
;;    pre-splitting alignments on both loop1 (even though the Propeller profile
;;    shows heavy fallthrough from entry and 0 backedge executions) and loop2.
; RUN: echo 'v1' > %t1
; RUN: echo 'f test_propeller' >> %t1
; RUN: echo 'c 0 2 4 5' >> %t1
; RUN: echo 'g 0:100,2:90,4:10 2:100,2:0,5:100 4:500,4:490,5:10 5:100' >> %t1
; RUN: llc < %s -O2 -mtriple=x86_64-pc-linux -function-sections -function-splitting=all \
; RUN:   -basic-block-sections=%t1 | FileCheck %s --check-prefixes=CHECK,SPLIT,NOALIGN
;;
;; 2. With -align-blocks-in-bbsections, alignment is deferred from MachineBlockPlacement
;;    to BasicBlockSections:
;;    - loop1 (heavy fallthrough 90/100 from layout predecessor entry) is not
;;      aligned;
;;    - loop2 (non-fallthrough after loop1, hot Freq=500) is aligned with
;;      -max-bytes-for-alignment=8;
;;    - test_pgo_only (PGO-only function splitting without a Propeller profile)
;;      has its loop header aligned via MBFIBlockAlignmentProfile.
; RUN: llc < %s -O2 -mtriple=x86_64-pc-linux -function-sections -function-splitting=all \
; RUN:   -basic-block-sections=%t1 -align-blocks-in-bbsections -max-bytes-for-alignment=8 \
; RUN:   | FileCheck %s --check-prefixes=CHECK,SPLIT,ALIGN
;;
;; 3. When a hot loop header (loop1) starts its own section (isBeginSection()),
;;    it has no fallthrough from its layout predecessor (entry, despite the hot
;;    90/100 edge) and is aligned.
; RUN: echo 'v1' > %t2
; RUN: echo 'f test_propeller' >> %t2
; RUN: echo 'c 0' >> %t2
; RUN: echo 'c 2 4 5' >> %t2
; RUN: echo 'g 0:100,2:90,4:10 2:100,2:0,5:100 4:500,4:490,5:10 5:100' >> %t2
; RUN: llc < %s -O2 -mtriple=x86_64-pc-linux -function-sections -function-splitting=all \
; RUN:   -basic-block-sections=%t2 -align-blocks-in-bbsections -max-bytes-for-alignment=8 \
; RUN:   | FileCheck %s --check-prefixes=CHECK,SPLIT,BEGINSEC
;;
;; 4. Even with a low entry count 2 < 5, an unexecuted loop is not aligned.
; RUN: echo 'v1' > %t3
; RUN: echo 'f test_propeller' >> %t3
; RUN: echo 'c 0 2 5 4' >> %t3
; RUN: echo 'g 0:2,2:2,4:0 2:2,2:0,5:2 4:0,4:0,5:0 5:2' >> %t3
; RUN: llc < %s -O2 -mtriple=x86_64-pc-linux -function-sections -function-splitting=all \
; RUN:   -basic-block-sections=%t3 -align-blocks-in-bbsections -max-bytes-for-alignment=8 \
; RUN:   | FileCheck %s --check-prefixes=CHECK,SPLIT,LOWENTRY
;;
;; 5. Without function splitting, the clusters still reorder the blocks within
;;    the function's single section and the same alignment applies (including
;;    for test_pgo_only, which is untouched by BasicBlockSections).
; RUN: llc < %s -O2 -mtriple=x86_64-pc-linux -function-sections -function-splitting=none \
; RUN:   -basic-block-sections=%t1 -align-blocks-in-bbsections -max-bytes-for-alignment=8 \
; RUN:   | FileCheck %s --check-prefixes=CHECK,NOSPLIT,ALIGN
;;
;; 6. With -basic-block-section-match-infer, inferred weights from
;;    BasicBlockMatchingAndInference are used for alignment.
; RUN: llc < %s -O2 -mtriple=x86_64-pc-linux -function-sections -function-splitting=all \
; RUN:   -basic-block-sections=%t1 -basic-block-section-match-infer -propeller-match-bbid \
; RUN:   -align-blocks-in-bbsections -max-bytes-for-alignment=8 \
; RUN:   | FileCheck %s --check-prefixes=CHECK,SPLIT,ALIGN
;;
;; 7. Enabling -align-blocks-in-bbsections without basic-block-sections (or
;;    function splitting / bb-addr-map) emits a warning and has no effect
;;    (MachineBlockPlacement still aligns blocks as usual).
; RUN: llc < %s -O2 -mtriple=x86_64-pc-linux -align-blocks-in-bbsections 2>&1 \
; RUN:   | FileCheck %s --check-prefixes=CHECK,NOSPLIT,NOALIGN,NOBBSECTIONS

; NOBBSECTIONS: warning: -align-blocks-in-bbsections has no effect when basic-block-sections is not enabled.

; CHECK-LABEL: test_propeller:
; CHECK:       # %bb.0:                                # %entry
; NOALIGN:     .p2align 4
; ALIGN-NOT:   .p2align
; LOWENTRY-NOT: .p2align
; NOALIGN:     .LBB0_1:                                # %loop1
; ALIGN:       .LBB0_1:                                # %loop1
; LOWENTRY:    .LBB0_1:                                # %loop1
; BEGINSEC:      .section .text.hot.test_propeller,"ax",@progbits,unique,1
; BEGINSEC-NEXT: .p2align 4, , 8
; BEGINSEC-NEXT: test_propeller.__part.1:              # %loop1

; NOALIGN:     .p2align 4
; ALIGN:       .p2align 4, , 8
; BEGINSEC:    .p2align 4, , 8
; NOALIGN-NEXT: .LBB0_2:                               # %loop2
; ALIGN-NEXT:   .LBB0_2:                               # %loop2
; BEGINSEC-NEXT: .LBB0_2:                              # %loop2

; LOWENTRY:      .LBB0_3:                              # %exit
; LOWENTRY-NOT:  .p2align
; LOWENTRY:      .LBB0_2:                              # %loop2

; CHECK-LABEL: test_pgo_only:
; CHECK:       # %bb.1:                                # %loop.preheader
; NOALIGN:     .p2align 4
; ALIGN:       .p2align 4, , 8
; BEGINSEC:    .p2align 4, , 8
; LOWENTRY:    .p2align 4, , 8
; CHECK-NEXT:  .LBB1_2:                                # %loop
; SPLIT:       .section .text.split.test_pgo_only,"ax",@progbits
; SPLIT-NEXT:  test_pgo_only.cold:
; NOSPLIT-NOT: test_pgo_only.cold:

define i32 @test_propeller(i32 %n, i1 %c) nounwind {
entry:
  br i1 %c, label %loop1, label %cold_pre

loop1:
  %i = phi i32 [ 0, %entry ], [ %inc, %loop1 ]
  call void @work()
  %inc = add i32 %i, 1
  %cmp = icmp slt i32 %inc, %n
  br i1 %cmp, label %loop1, label %exit

cold_pre:
  br label %loop2

loop2:
  %j = phi i32 [ 0, %cold_pre ], [ %jinc, %loop2 ]
  call void @work()
  %jinc = add i32 %j, 1
  %jcmp = icmp slt i32 %jinc, %n
  br i1 %jcmp, label %loop2, label %exit

exit:
  ret i32 0
}

define i32 @test_pgo_only(i32 %n, i1 %c) nounwind !prof !14 {
entry:
  br i1 %c, label %cold, label %loop, !prof !15

cold:
  call void @work()
  br label %exit

loop:
  %i = phi i32 [ 0, %entry ], [ %inc, %loop ]
  call void @work()
  %inc = add i32 %i, 1
  %cmp = icmp slt i32 %inc, %n
  br i1 %cmp, label %loop, label %exit, !prof !16

exit:
  ret i32 0
}

declare void @work()

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"ProfileSummary", !1}
!1 = !{!2, !3, !4, !5, !6, !7, !8, !9}
!2 = !{!"ProfileFormat", !"InstrProf"}
!3 = !{!"TotalCount", i64 10000}
!4 = !{!"MaxCount", i64 10}
!5 = !{!"MaxInternalCount", i64 1}
!6 = !{!"MaxFunctionCount", i64 1000}
!7 = !{!"NumCounts", i64 3}
!8 = !{!"NumFunctions", i64 5}
!9 = !{!"DetailedSummary", !10}
!10 = !{!11, !12, !13}
!11 = !{i32 10000, i64 100, i32 1}
!12 = !{i32 999900, i64 100, i32 1}
!13 = !{i32 999999, i64 1, i32 2}
!14 = !{!"function_entry_count", i64 7000}
!15 = !{!"branch_weights", i32 0, i32 7000}
!16 = !{!"branch_weights", i32 6999, i32 1}
