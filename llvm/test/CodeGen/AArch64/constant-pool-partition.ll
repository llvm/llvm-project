; RUN: llc -mtriple=aarch64 -partition-static-data-sections \
; RUN:     -function-sections -unique-section-names=false \
; RUN:     %s -o - 2>&1 | FileCheck %s --dump-input=always

; Repeat the RUN command above for big-endian systems.
; RUN: llc -mtriple=aarch64_be -partition-static-data-sections \
; RUN:     -function-sections -unique-section-names=false \
; RUN:     %s -o - 2>&1 | FileCheck %s --dump-input=always

; Tests that constant pool hotness is aggregated across the module. The
; static-data-splitter processes data from cold_func first, unprofiled_func
; secondly, and then hot_func. Specifically, tests that
; - If a constant is accessed by hot functions, all constant pools for this
;   constant (e.g., from an unprofiled function, or cold function) should have
;   `.hot` suffix. For instance, double 0.68 is seen by both @cold_func and
;   @hot_func, so two CPI emits (under label LCPI0_0 and LCPI2_0) have `.hot`
;   suffix.
; - Similarly if a constant is accessed by both cold function and un-profiled
;   function, constant pools for this constant should not have `.unlikely` suffix.

;; Constant pools for function @cold_func.
; CHECK:       .section	.rodata.cst8.hot,"aM",@progbits,8
; CHECK-NEXT:     .p2align
; CHECK-NEXT:   .LCPI0_0:
; CHECK-NEXT:	    .xword	0x3fe5c28f5c28f5c3              // double 0.68000000000000005
; CHECK-NEXT: .section	.rodata.cst8.unlikely,"aM",@progbits,8
; CHECK-NEXT:     .p2align
; CHECK-NEXT:   .LCPI0_1:
; CHECK-NEXT:     .xword 0x3fe5eb851eb851ec              // double 0.68500000000000005
; CHECK-NEXT:	.section	.rodata.cst8,"aM",@progbits,8
; CHECK-NEXT:     .p2align
; CHECK-NEXT:   .LCPI0_2:
; CHECK-NEXT:     .byte   0                               // 0x0
; CHECK-NEXT:     .byte   4                               // 0x4
; CHECK-NEXT:     .byte   8                               // 0x8
; CHECK-NEXT:     .byte   12                              // 0xc
; CHECK-NEXT:     .byte   255                             // 0xff
; CHECK-NEXT:     .byte   255                             // 0xff
; CHECK-NEXT:     .byte   255                             // 0xff
; CHECK-NEXT:     .byte   255                             // 0xff

;; Constant pools for function @unprofiled_func
; CHECK:	    .section	.rodata.cst8,"aM",@progbits,8
; CHECK-NEXT:     .p2align
; CHECK-NEXT:   .LCPI1_0:
; CHECK-NEXT:     .byte   0                               // 0x0
; CHECK-NEXT:     .byte   4                               // 0x4
; CHECK-NEXT:     .byte   8                               // 0x8
; CHECK-NEXT:     .byte   12                              // 0xc
; CHECK-NEXT:     .byte   255                             // 0xff
; CHECK-NEXT:     .byte   255                             // 0xff
; CHECK-NEXT:     .byte   255                             // 0xff
; CHECK-NEXT:     .byte   255                             // 0xff
; CHECK-NEXT: .section .rodata.cst16,"aM",@progbits,16
; CHECK-NEXT:     .p2align
; CHECK-NEXT:   .LCPI1_1:
; CHECK-NEXT:     .word 2                                 // 0x2
; CHECK-NEXT:     .word 3                                 // 0x3
; CHECK-NEXT:     .word 5                                 // 0x5
; CHECK-NEXT:     .word 7                                 // 0x7
; CHECK-NEXT: .section        .rodata.cst16.hot,"aM",@progbits,16
; CHECK-NEXT:     .p2align
; CHECK-NEXT:   .LCPI1_2:
; CHECK-NEXT:     .word   442                             // 0x1ba
; CHECK-NEXT:     .word   100                             // 0x64
; CHECK-NEXT:     .word   0                               // 0x0
; CHECK-NEXT:     .word   0                               // 0x0

;; Constant pools for function @hot_func
; CHECK:      .section        .rodata.cst8.hot,"aM",@progbits,8
; CHECK-NEXT:     .p2align
; CHECK-NEXT:   .LCPI2_0:
; CHECK-NEXT:     .xword  0x3fe5c28f5c28f5c3              // double 0.68000000000000005
; CHECK-NEXT: .section        .rodata.cst16.hot,"aM",@progbits,16
; CHECK-NEXT:     .p2align
; CHECK-NEXT:   .LCPI2_1:
; CHECK-NEXT:     .word   0                               // 0x0
; CHECK-NEXT:     .word   100                             // 0x64
; CHECK-NEXT:     .word   0                               // 0x0
; CHECK-NEXT:     .word   442                             // 0x1ba
; CHECK-NEXT:   .LCPI2_2:
; CHECK-NEXT:     .word   442                             // 0x1ba
; CHECK-NEXT:     .word   100                             // 0x64
; CHECK-NEXT:     .word   0                               // 0x0
; CHECK-NEXT:     .word   0                               // 0x0

;; For global variable @val
;; The section name remains `.rodata.cst32` without hotness prefix because
;; the variable has external linkage and not analyzed. Compiler need symbolized
;; data access profiles to annotate such global variables' hotness.
; CHECK:       .section	.rodata.cst32,"aM",@progbits,32
; CHECK-NEXT:  .globl	val

define i32 @cold_func(double %x, <16 x i8> %a, <16 x i8> %b) !prof !16 {
  %2 = tail call i32 (...) @func_taking_arbitrary_param(double 6.800000e-01)
  %num = tail call i32 (...) @func_taking_arbitrary_param(double 6.8500000e-01)
  %t1 = call <8 x i8> @llvm.aarch64.neon.tbl2.v8i8(<16 x i8> %a, <16 x i8> %b, <8 x i8> <i8 0, i8 4, i8 8, i8 12, i8 -1, i8 -1, i8 -1, i8 -1>)
  %t2 = bitcast <8 x i8> %t1 to <2 x i32>
  %3 = extractelement <2 x i32> %t2, i32 1
  %sum = add i32 %2, %3
  %ret = add i32 %sum, %num
  ret i32 %ret
}

declare <8 x i8> @llvm.aarch64.neon.tbl2.v8i8(<16 x i8>, <16 x i8>, <8 x i8>)
declare i32 @func_taking_arbitrary_param(...)

define <4 x i1> @unprofiled_func(<16 x i8> %a, <16 x i8> %b) {
  %t1 = call <8 x i8> @llvm.aarch64.neon.tbl2.v8i8(<16 x i8> %a, <16 x i8> %b, <8 x i8> <i8 0, i8 4, i8 8, i8 12, i8 -1, i8 -1, i8 -1, i8 -1>)
  %t2 = bitcast <8 x i8> %t1 to <4 x i16>
  %t3 = zext <4 x i16> %t2 to <4 x i32>
  %t4 = add <4 x i32> %t3, <i32 2, i32 3, i32 5, i32 7>
  %cmp = icmp ule <4 x i32> <i32 442, i32 100, i32 0, i32 0>, %t4
  ret <4 x i1> %cmp
}

define <4 x i1> @hot_func(i32 %0, <4 x i32> %a) !prof !17 {
  %2 = tail call i32 (...) @func_taking_arbitrary_param(double 6.800000e-01)
  %b = add <4 x i32> <i32 0, i32 100, i32 0, i32 442>, %a
  %c = icmp ule <4 x i32> %b, <i32 442, i32 100, i32 0, i32 0>
  ret <4 x i1> %c
}

@val = unnamed_addr constant i256 1

define i32 @main(i32 %0, ptr %1) !prof !16 {
  br label %7

5:                                                ; preds = %7
  %x = call double @double_func()
  %a = call <16 x i8> @vector_func_16i8()
  %b = call <16 x i8> @vector_func_16i8()
  call void @cold_func(double %x, <16 x i8> %a, <16 x i8> %b)
  ret i32 0

7:                                                ; preds = %7, %2
  %8 = phi i32 [ 0, %2 ], [ %10, %7 ]
  %seed_val = load i256, ptr @val
  %9 = call i32 @seed(i256 %seed_val)
  call void @hot_func(i32 %9)
  %10 = add i32 %8, 1
  %11 = icmp eq i32 %10, 100000
  br i1 %11, label %5, label %7, !prof !18
}

declare i32 @seed(i256)
declare double @double_func()
declare <4 x i32> @vector_func()
declare <16 x i8> @vector_func_16i8()

!llvm.module.flags = !{!1}

!1 = !{i32 1, !"ProfileSummary", !2}
!2 = !{!3, !4, !5, !6, !7, !8, !9, !10, !11, !12}
!3 = !{!"ProfileFormat", !"InstrProf"}
!4 = !{!"TotalCount", i64 1460617}
!5 = !{!"MaxCount", i64 849536}
!6 = !{!"MaxInternalCount", i64 32769}
!7 = !{!"MaxFunctionCount", i64 849536}
!8 = !{!"NumCounts", i64 23784}
!9 = !{!"NumFunctions", i64 3301}
!10 = !{!"IsPartialProfile", i64 0}
!11 = !{!"PartialProfileRatio", double 0.000000e+00}
!12 = !{!"DetailedSummary", !13}
!13 = !{!14, !15}
!14 = !{i32 990000, i64 166, i32 73}
!15 = !{i32 999999, i64 3, i32 1463}
!16 = !{!"function_entry_count", i64 1}
!17 = !{!"function_entry_count", i64 100000}
!18 = !{!"branch_weights", i32 1, i32 99999}
