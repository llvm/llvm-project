; RUN: llc -O0 < %s | FileCheck %s --check-prefixes=CHECK,DEFCM
; RUN: llc -O1 < %s | FileCheck %s --check-prefixes=CHECK,DEFCM
; RUN: llc -O2 < %s | FileCheck %s --check-prefixes=CHECK,DEFCM
; RUN: llc -O3 < %s | FileCheck %s --check-prefixes=CHECK,DEFCM
; RUN: llc -O1 -code-model=large < %s | FileCheck %s --check-prefixes=CHECK,LARGE

target triple = "x86_64-unknown-linux-gnu"

@foo = dso_local global i64 0, align 8
@bar = dso_local global i64 0, align 8

define void @empty_no_aux() !pcsections !0 {
; CHECK-LABEL: empty_no_aux:
; CHECK-NEXT:  .Lfunc_begin0
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    retq
; CHECK-NEXT:  .Lfunc_end0:
; CHECK:       .section	section_no_aux,"awo",@progbits,.text
; CHECK-NEXT:  .Lpcsection_base0:
; DEFCM-NEXT:  .long	.Lfunc_begin0-.Lpcsection_base0
; LARGE-NEXT:  .quad	.Lfunc_begin0-.Lpcsection_base0
; CHECK-NEXT:  .long	.Lfunc_end0-.Lfunc_begin0
; CHECK-NEXT:  .text
entry:
  ret void
}

define void @empty_aux() !pcsections !1 {
; CHECK-LABEL: empty_aux:
; CHECK-NEXT:  .Lfunc_begin1
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    retq
; CHECK-NEXT:  .Lfunc_end1:
; CHECK:       .section	section_aux,"awo",@progbits,.text
; CHECK-NEXT:  .Lpcsection_base1:
; DEFCM-NEXT:  .long	.Lfunc_begin1-.Lpcsection_base1
; LARGE-NEXT:  .quad	.Lfunc_begin1-.Lpcsection_base1
; CHECK-NEXT:  .long	.Lfunc_end1-.Lfunc_begin1
; CHECK-NEXT:  .long	10
; CHECK-NEXT:  .long	20
; CHECK-NEXT:  .long	30
; CHECK-NEXT:  .text
entry:
  ret void
}

define i64 @multiple() !pcsections !0 {
; CHECK-LABEL: multiple:
; CHECK-NEXT:  .Lfunc_begin2
; CHECK:       # %bb.0: # %entry
; CHECK:       .Lpcsection0:
; CHECK-NEXT:    movq
; CHECK-NEXT:    retq
; CHECK-NEXT:  .Lfunc_end2:
; CHECK:       .section	section_no_aux,"awo",@progbits,.text
; CHECK-NEXT:  .Lpcsection_base2:
; DEFCM-NEXT:  .long	.Lfunc_begin2-.Lpcsection_base2
; LARGE-NEXT:  .quad	.Lfunc_begin2-.Lpcsection_base2
; CHECK-NEXT:  .long	.Lfunc_end2-.Lfunc_begin2
; CHECK-NEXT:  .section	section_aux_42,"awo",@progbits,.text
; CHECK-NEXT:  .Lpcsection_base3:
; DEFCM-NEXT:  .long	.Lpcsection0-.Lpcsection_base3
; LARGE-NEXT:  .quad	.Lpcsection0-.Lpcsection_base3
; CHECK-NEXT:  .long	42
; CHECK-NEXT:  .section	section_aux_21264,"awo",@progbits,.text
; CHECK-NEXT:  .Lpcsection_base4:
; DEFCM-NEXT:  .long	.Lpcsection0-.Lpcsection_base4
; LARGE-NEXT:  .quad	.Lpcsection0-.Lpcsection_base4
; CHECK-NEXT:  .long	21264
; CHECK-NEXT:  .text
entry:
  %0 = load i64, i64* @bar, align 8, !pcsections !2
  ret i64 %0
}

define i64 @test_simple_atomic() {
; CHECK-LABEL: test_simple_atomic:
; CHECK:       .Lpcsection1:
; CHECK-NEXT:    movq
; CHECK-NOT:   .Lpcsection
; CHECK:         addq
; CHECK-NEXT:    retq
; CHECK-NEXT:  .Lfunc_end3:
; CHECK:       .section	section_no_aux,"awo",@progbits,.text
; CHECK-NEXT:  .Lpcsection_base5:
; DEFCM-NEXT:  .long	.Lpcsection1-.Lpcsection_base5
; LARGE-NEXT:  .quad	.Lpcsection1-.Lpcsection_base5
; CHECK-NEXT:  .text
entry:
  %0 = load atomic i64, i64* @foo monotonic, align 8, !pcsections !0
  %1 = load i64, i64* @bar, align 8
  %add = add nsw i64 %1, %0
  ret i64 %add
}

define i64 @test_complex_atomic() {
; CHECK-LABEL: test_complex_atomic:
; CHECK:         movl $1
; CHECK-NEXT:  .Lpcsection2:
; CHECK-NEXT:    lock xaddq
; CHECK-NOT:   .Lpcsection
; CHECK:         movq
; CHECK:         addq
; CHECK:         retq
; CHECK-NEXT:  .Lfunc_end4:
; CHECK:       .section	section_no_aux,"awo",@progbits,.text
; CHECK-NEXT:  .Lpcsection_base6:
; DEFCM-NEXT:  .long	.Lpcsection2-.Lpcsection_base6
; LARGE-NEXT:  .quad	.Lpcsection2-.Lpcsection_base6
; CHECK-NEXT:  .text
entry:
  %0 = atomicrmw add i64* @foo, i64 1 monotonic, align 8, !pcsections !0
  %1 = load i64, i64* @bar, align 8
  %inc = add nsw i64 %1, 1
  store i64 %inc, i64* @bar, align 8
  %add = add nsw i64 %1, %0
  ret i64 %add
}

!0 = !{!"section_no_aux"}
!1 = !{!"section_aux", !3}
!2 = !{!"section_aux_42", !4, !"section_aux_21264", !5}
!3 = !{i32 10, i32 20, i32 30}
!4 = !{i32 42}
!5 = !{i32 21264}
