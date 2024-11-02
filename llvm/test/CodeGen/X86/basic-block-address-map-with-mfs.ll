; COM: Emitting basic-block-address-map when machine function splitting is enabled.
; RUN: llc < %s -mtriple=x86_64 -function-sections -split-machine-functions -basic-block-address-map | FileCheck %s --check-prefixes=CHECK,BASIC

; COM: Emitting basic-block-address-map with PGO analysis with machine function splitting enabled.
; RUN: llc < %s -mtriple=x86_64 -function-sections -split-machine-functions -basic-block-address-map -pgo-analysis-map=func-entry-count,bb-freq,br-prob | FileCheck %s --check-prefixes=CHECK,PGO

define void @foo(i1 zeroext %0) nounwind !prof !14 {
  br i1 %0, label %2, label %4, !prof !15

2:                                                ; preds = %1
  %3 = call i32 @bar()
  br label %6

4:                                                ; preds = %1
  %5 = call i32 @baz()
  br label %6

6:                                                ; preds = %4, %2
  %7 = tail call i32 @qux()
  ret void
}

declare i32 @bar()
declare i32 @baz()
declare i32 @qux()

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
!15 = !{!"branch_weights", i32 7000, i32 0}

; CHECK:          .section .text.hot.foo,"ax",@progbits
; CHECK-LABEL:  foo:
; CHECK-LABEL:  .Lfunc_begin0:
; CHECK-LABEL:  .LBB_END0_0:
; CHECK-LABEL:  .LBB0_1:
; CHECK-LABEL:  .LBB_END0_1:
; CHECK:          .section .text.split.foo,"ax",@progbits
; CHECK-LABEL:  foo.cold:
; CHECK-LABEL:  .LBB_END0_2:
; CHECK-LABEL:  .Lfunc_end0:

; CHECK:                .section        .llvm_bb_addr_map,"o",@llvm_bb_addr_map,.text.hot.foo
; CHECK-NEXT:   .byte   2               # version
; BASIC-NEXT:   .byte   8               # feature
; PGO-NEXT:     .byte   15              # feature
; CHECK-NEXT:   .byte   2               # number of basic block ranges
; CHECK-NEXT:   .quad   .Lfunc_begin0   # base address
; CHECK-NEXT:   .byte   2               # number of basic blocks
; CHECK-NEXT:   .byte   0               # BB id
; CHECK-NEXT:   .uleb128 .Lfunc_begin0-.Lfunc_begin0
; CHECK-NEXT:   .uleb128 .LBB_END0_0-.Lfunc_begin0
; CHECK-NEXT:   .byte   8
; CHECK-NEXT:   .byte   1               # BB id
; CHECK-NEXT:   .uleb128 .LBB0_1-.LBB_END0_0
; CHECK-NEXT:   .uleb128 .LBB_END0_1-.LBB0_1
; CHECK-NEXT:   .byte   3
; CHECK-NEXT:   .quad   foo.cold    # base address
; CHECK-NEXT:   .byte   1               # number of basic blocks
; CHECK-NEXT:   .byte   2               # BB id
; CHECK-NEXT:   .uleb128 foo.cold-foo.cold
; CHECK-NEXT:   .uleb128 .LBB_END0_2-foo.cold
; CHECK-NEXT:   .byte   3

;; PGO Analysis Map
; PGO:         .ascii  "\3306"                            # function entry count
; PGO-NEXT:    .ascii  "\200\200\200\200\200\200\200 "    # basic block frequency
; PGO-NEXT:    .byte   2                                  # basic block successor count
; PGO-NEXT:    .byte   1                                  # successor BB ID
; PGO-NEXT:    .ascii  "\200\200\200\200\b"               # successor branch probability
; PGO-NEXT:    .byte   2                                  # successor BB ID
; PGO-NEXT:    .byte   0                                  # successor branch probability
; PGO-NEXT:    .ascii  "\200\200\200\374\377\377\377\037" # basic block frequency
; PGO-NEXT:    .byte   0		                       # basic block successor count
; PGO-NEXT:    .ascii  "\200\200\200\004"                 # basic block frequency
; PGO-NEXT:    .byte   0                                  # basic block successor count
