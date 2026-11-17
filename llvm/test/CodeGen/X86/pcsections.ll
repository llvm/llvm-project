; This test should verify that adding !pcsections metadata is encoded properly
; by the AsmPrinter. For tests to check that metadata is propagated to
; assembly, see pcsections-*.ll tests.

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
; CHECK:       .section	section_no_aux,"awo",@progbits,.{{l?}}text
; CHECK-NEXT:  .Lpcsection_base0:
; DEFCM-NEXT:  .long	.Lfunc_begin0-.Lpcsection_base0
; LARGE-NEXT:  .quad	.Lfunc_begin0-.Lpcsection_base0
; CHECK-NEXT:  .long	.Lfunc_end0-.Lfunc_begin0
; CHECK-NEXT:  .{{l?}}text
entry:
  ret void
}

define void @empty_aux() !pcsections !1 {
; CHECK-LABEL: empty_aux:
; CHECK-NEXT:  .Lfunc_begin1
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    retq
; CHECK-NEXT:  .Lfunc_end1:
; CHECK:       .section	section_aux,"awo",@progbits,.{{l?}}text
; CHECK-NEXT:  .Lpcsection_base1:
; DEFCM-NEXT:  .long	.Lfunc_begin1-.Lpcsection_base1
; LARGE-NEXT:  .quad	.Lfunc_begin1-.Lpcsection_base1
; CHECK-NEXT:  .long	.Lfunc_end1-.Lfunc_begin1
; CHECK-NEXT:  .long	10
; CHECK-NEXT:  .long	20
; CHECK-NEXT:  .long	30
; CHECK-NEXT:  .{{l?}}text
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
; CHECK:       .section	section_no_aux,"awo",@progbits,.{{l?}}text
; CHECK-NEXT:  .Lpcsection_base2:
; DEFCM-NEXT:  .long	.Lfunc_begin2-.Lpcsection_base2
; LARGE-NEXT:  .quad	.Lfunc_begin2-.Lpcsection_base2
; CHECK-NEXT:  .long	.Lfunc_end2-.Lfunc_begin2
; CHECK-NEXT:  .section	section_aux_42,"awo",@progbits,.{{l?}}text
; CHECK-NEXT:  .Lpcsection_base3:
; DEFCM-NEXT:  .long	.Lpcsection0-.Lpcsection_base3
; LARGE-NEXT:  .quad	.Lpcsection0-.Lpcsection_base3
; CHECK-NEXT:  .long	42
; CHECK-NEXT:  .section	section_aux_21264,"awo",@progbits,.{{l?}}text
; CHECK-NEXT:  .Lpcsection_base4:
; DEFCM-NEXT:  .long	.Lpcsection0-.Lpcsection_base4
; LARGE-NEXT:  .quad	.Lpcsection0-.Lpcsection_base4
; CHECK-NEXT:  .long	21264
; CHECK-NEXT:  .{{l?}}text
entry:
  %0 = load i64, ptr @bar, align 8, !pcsections !2
  ret i64 %0
}

define void @multiple_uleb128() !pcsections !6 {
; CHECK-LABEL: multiple_uleb128:
; CHECK:       .section	section_aux,"awo",@progbits,.{{l?}}text
; CHECK-NEXT:  .Lpcsection_base5:
; DEFCM-NEXT:  .long	.Lfunc_begin3-.Lpcsection_base5
; LARGE-NEXT:  .quad	.Lfunc_begin3-.Lpcsection_base5
; CHECK-NEXT:  .uleb128	.Lfunc_end3-.Lfunc_begin3
; CHECK-NEXT:  .byte	42
; CHECK-NEXT:  .ascii	"\345\216&"
; CHECK-NEXT:  .byte	255
; CHECK-NEXT:  .section	section_aux_21264,"awo",@progbits,.{{l?}}text
; CHECK-NEXT:  .Lpcsection_base6:
; DEFCM-NEXT:  .long	.Lfunc_begin3-.Lpcsection_base6
; LARGE-NEXT:  .quad	.Lfunc_begin3-.Lpcsection_base6
; CHECK-NEXT:  .long	.Lfunc_end3-.Lfunc_begin3
; CHECK-NEXT:  .long	21264
; CHECK-NEXT:  .{{l?}}text
entry:
  ret void
}

; Instructions sharing one MDNode emit one complete record each, in program
; order: PC, then the auxiliary data.
define void @multiple_insts_same_md() {
; CHECK-LABEL: multiple_insts_same_md:
; CHECK:       [[PC1:.Lpcsection[0-9]+]]:
; CHECK-NEXT:    movq $1,
; CHECK:       [[PC2:.Lpcsection[0-9]+]]:
; CHECK-NEXT:    movq $2,
; CHECK-NEXT:    retq
; CHECK:       .section	section_shared,"awo",@progbits,.{{l?}}text
; CHECK-NEXT:  [[BASE1:.Lpcsection_base[0-9]+]]:
; DEFCM-NEXT:  .long	[[PC1]]-[[BASE1]]
; LARGE-NEXT:  .quad	[[PC1]]-[[BASE1]]
; CHECK-NEXT:  .byte	42
; CHECK-NEXT:  .ascii	"\345\216&"
; CHECK-NEXT:  .byte	255
; CHECK-NEXT:  [[BASE2:.Lpcsection_base[0-9]+]]:
; DEFCM-NEXT:  .long	[[PC2]]-[[BASE2]]
; LARGE-NEXT:  .quad	[[PC2]]-[[BASE2]]
; CHECK-NEXT:  .byte	42
; CHECK-NEXT:  .ascii	"\345\216&"
; CHECK-NEXT:  .byte	255
; CHECK-NEXT:  .{{l?}}text
entry:
  store volatile i64 1, ptr @foo, align 8, !pcsections !8
  store volatile i64 2, ptr @bar, align 8, !pcsections !8
  ret void
}

; A node that requests one section twice emits two records for the PC, each
; with its own auxiliary data.
define void @merged_requests() {
; CHECK-LABEL: merged_requests:
; CHECK:       [[PC:.Lpcsection[0-9]+]]:
; CHECK:       .section	section_merged,"awo",@progbits,.{{l?}}text
; CHECK-NEXT:  [[BASE4:.Lpcsection_base[0-9]+]]:
; DEFCM-NEXT:  .long	[[PC]]-[[BASE4]]
; LARGE-NEXT:  .quad	[[PC]]-[[BASE4]]
; CHECK-NEXT:  .long	42
; CHECK-NEXT:  [[BASE5:.Lpcsection_base[0-9]+]]:
; DEFCM-NEXT:  .long	[[PC]]-[[BASE5]]
; LARGE-NEXT:  .quad	[[PC]]-[[BASE5]]
; CHECK-NEXT:  .long	21264
; CHECK-NEXT:  .{{l?}}text
entry:
  store volatile i64 1, ptr @foo, align 8, !pcsections !9
  ret void
}

!0 = !{!"section_no_aux"}
!1 = !{!"section_aux", !3}
!2 = !{!"section_aux_42", !4, !"section_aux_21264", !5}
!3 = !{i32 10, i32 20, i32 30}
!4 = !{i32 42}
!5 = !{i32 21264}
!6 = !{!"section_aux!C", !7, !"section_aux_21264", !5}
!7 = !{i64 42, i32 624485, i8 255}
!8 = !{!"section_shared!C", !7}
!9 = !{!"section_merged", !4, !"section_merged", !5}
