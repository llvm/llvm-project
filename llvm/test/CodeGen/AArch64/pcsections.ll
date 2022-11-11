; RUN: llc -O0 < %s | FileCheck %s --check-prefixes=CHECK,CHECK-UNOPT,DEFCM
; RUN: llc -O1 < %s | FileCheck %s --check-prefixes=CHECK,CHECK-OPT,DEFCM
; RUN: llc -O2 < %s | FileCheck %s --check-prefixes=CHECK,CHECK-OPT,DEFCM
; RUN: llc -O3 < %s | FileCheck %s --check-prefixes=CHECK,CHECK-OPT,DEFCM
; RUN: llc -O1 -code-model=large < %s | FileCheck %s --check-prefixes=CHECK,CHECK-OPT,LARGE

target triple = "aarch64-unknown-linux-gnu"

@foo = dso_local global i64 0, align 8
@bar = dso_local global i64 0, align 8

define i64 @multiple() !pcsections !0 {
; CHECK-LABEL: multiple:
; CHECK:       .Lfunc_begin0:
; CHECK:       // %bb.0: // %entry
; CHECK:       .Lpcsection0:
; CHECK-NEXT:    ldr
; CHECK-NEXT:    ret
; CHECK:       .section	section_no_aux,"awo",@progbits,.text
; CHECK-NEXT:  .Lpcsection_base0:
; DEFCM-NEXT:  .word	.Lfunc_begin0-.Lpcsection_base0
; LARGE-NEXT:  .xword	.Lfunc_begin0-.Lpcsection_base0
; CHECK-NEXT:  .word	.Lfunc_end0-.Lfunc_begin0
; CHECK-NEXT:  .section	section_aux_42,"awo",@progbits,.text
; CHECK-NEXT:  .Lpcsection_base1:
; DEFCM-NEXT:  .word	.Lpcsection0-.Lpcsection_base1
; LARGE-NEXT:  .xword	.Lpcsection0-.Lpcsection_base1
; CHECK-NEXT:  .word	42
; CHECK-NEXT:  .section	section_aux_21264,"awo",@progbits,.text
; CHECK-NEXT:  .Lpcsection_base2:
; DEFCM-NEXT:  .word	.Lpcsection0-.Lpcsection_base2
; LARGE-NEXT:  .xword	.Lpcsection0-.Lpcsection_base2
; CHECK-NEXT:  .word	21264
; CHECK-NEXT:  .text
entry:
  %0 = load i64, i64* @bar, align 8, !pcsections !1
  ret i64 %0
}

define i64 @test_simple_atomic() {
; CHECK-LABEL: test_simple_atomic:
; CHECK:       .Lpcsection1:
; CHECK-NEXT:    ldr
; CHECK-NOT:   .Lpcsection2
; CHECK:         ldr
; CHECK:         add
; CHECK-NEXT:    ret
; CHECK:       .section	section_no_aux,"awo",@progbits,.text
; CHECK-NEXT:  .Lpcsection_base3:
; DEFCM-NEXT:  .word	.Lpcsection1-.Lpcsection_base3
; LARGE-NEXT:  .xword	.Lpcsection1-.Lpcsection_base3
; CHECK-NEXT:  .text
entry:
  %0 = load atomic i64, i64* @foo monotonic, align 8, !pcsections !0
  %1 = load i64, i64* @bar, align 8
  %add = add nsw i64 %1, %0
  ret i64 %add
}

define i64 @test_complex_atomic() {
; CHECK-LABEL: test_complex_atomic:
; ---
; CHECK-OPT:       .Lpcsection2:
; CHECK-OPT-NEXT:    ldxr
; CHECK-OPT:       .Lpcsection3:
; CHECK-OPT-NEXT:    add
; CHECK-OPT:       .Lpcsection4:
; CHECK-OPT-NEXT:    stxr
; CHECK-OPT:       .Lpcsection5:
; CHECK-OPT-NEXT:    cbnz
; ---
; CHECK-UNOPT:     .Lpcsection2:
; CHECK-UNOPT-NEXT:  ldr
; CHECK-UNOPT:     .Lpcsection4:
; CHECK-UNOPT-NEXT:  add
; CHECK-UNOPT:     .Lpcsection5:
; CHECK-UNOPT-NEXT:  ldaxr
; CHECK-UNOPT:     .Lpcsection6:
; CHECK-UNOPT-NEXT:  cmp
; CHECK-UNOPT:     .Lpcsection8:
; CHECK-UNOPT-NEXT:  stlxr
; CHECK-UNOPT:     .Lpcsection9:
; CHECK-UNOPT-NEXT:  cbnz
; CHECK-UNOPT:     .Lpcsection13:
; CHECK-UNOPT-NEXT:  b
; ---
; CHECK-NOT:   .Lpcsection
; CHECK:         ldr
; CHECK:         ret
; CHECK:       .section	section_no_aux,"awo",@progbits,.text
; CHECK-NEXT:  .Lpcsection_base4:
; DEFCM-NEXT:  .word	.Lpcsection2-.Lpcsection_base4
; LARGE-NEXT:  .xword	.Lpcsection2-.Lpcsection_base4
; CHECK-NEXT:  .Lpcsection_base5:
; DEFCM-NEXT:  .word	.Lpcsection3-.Lpcsection_base5
; LARGE-NEXT:  .xword	.Lpcsection3-.Lpcsection_base5
; CHECK-NEXT:  .Lpcsection_base6:
; DEFCM-NEXT:  .word	.Lpcsection4-.Lpcsection_base6
; LARGE-NEXT:  .xword	.Lpcsection4-.Lpcsection_base6
; CHECK-NEXT:  .Lpcsection_base7:
; DEFCM-NEXT:  .word	.Lpcsection5-.Lpcsection_base7
; LARGE-NEXT:  .xword	.Lpcsection5-.Lpcsection_base7
; CHECK-UNOPT: .word	.Lpcsection13-.Lpcsection_base15
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
!1 = !{!"section_aux_42", !2, !"section_aux_21264", !3}
!2 = !{i32 42}
!3 = !{i32 21264}
