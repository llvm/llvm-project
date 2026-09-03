; RUN: llc < %s -relocation-model=pic -code-model=medium -large-data-threshold=0 | FileCheck %s --check-prefix=LARGE --implicit-check-not=rodata
; RUN: llc < %s -relocation-model=pic -code-model=medium -large-data-threshold=8 | FileCheck %s --check-prefix=MIXED --implicit-check-not=rodata
; RUN: llc < %s -relocation-model=pic -code-model=medium -large-data-threshold=100 | FileCheck %s --check-prefix=SMALL --implicit-check-not=rodata

; RUN: llc < %s -relocation-model=pic -code-model=medium -large-data-threshold=0 \
; RUN:     -partition-static-data-sections | FileCheck %s --check-prefix=SUFFIX-LARGE --implicit-check-not=rodata
; RUN: llc < %s -relocation-model=pic -code-model=medium -large-data-threshold=8 \
; RUN:     -partition-static-data-sections | FileCheck %s --check-prefix=SUFFIX-MIXED --implicit-check-not=rodata
; RUN: llc < %s -relocation-model=pic -code-model=medium -large-data-threshold=100 \
; RUN:     -partition-static-data-sections | FileCheck %s --check-prefix=SUFFIX-SMALL --implicit-check-not=rodata

; LARGE: .section .lrodata.str1.1,"aMSl",@progbits,1

; MIXED: .section .lrodata.str1.1,"aMSl",@progbits,1
; MIXED: .section .rodata.str1.1,"aMS",@progbits,1

; SMALL: .section .rodata.str1.1,"aMS",@progbits,1

; SUFFIX-LARGE: .section .lrodata.str1.1.hot.,"aMSl",@progbits,1

; SUFFIX-MIXED: .section .lrodata.str1.1.hot.,"aMSl",@progbits,1
; SUFFIX-MIXED: .section .rodata.str1.1.hot.,"aMS",@progbits,1

; SUFFIX-SMALL: .section .rodata.str1.1.hot.,"aMS",@progbits,1

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64--linux"

@.str_large = private unnamed_addr constant [17 x i8] c"0123456789abcdef\00", align 1
@.str_small = private unnamed_addr constant [4 x i8] c"abc\00", align 1

define ptr @get_str_large() !prof !17 {
  ret ptr @.str_large
}

define ptr @get_str_small() !prof !17 {
  ret ptr @.str_small
}

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
!15 = !{i32 999999, i64 1, i32 1463}
!16 = !{!"function_entry_count", i64 1}
!17 = !{!"function_entry_count", i64 100000}
