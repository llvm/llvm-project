; RUN: llc -O0 < %s | FileCheck %s --check-prefixes=CHECK,CHECK-UNOPT,DEFCM
; RUN: llc -O1 < %s | FileCheck %s --check-prefixes=CHECK,CHECK-OPT,DEFCM
; RUN: llc -O2 < %s | FileCheck %s --check-prefixes=CHECK,CHECK-OPT,DEFCM
; RUN: llc -O3 < %s | FileCheck %s --check-prefixes=CHECK,CHECK-OPT,DEFCM
; RUN: llc -O1 -code-model=large < %s | FileCheck %s --check-prefixes=CHECK,LARGE

target triple = "aarch64-unknown-linux-gnu"

@foo = dso_local global i64 0, align 8
@bar = dso_local global i64 0, align 8

define i64 @multiple() !pcsections !0 {
; CHECK-LABEL: multiple:
; CHECK:       .Lfunc_begin0:
; CHECK:       // %bb.0: // %entry
; CHECK:       .Lpcsection0:
; --
; LARGE-NEXT:  movz
; LARGE:  .Lpcsection1:
; LARGE-NEXT:  movk
; LARGE:  .Lpcsection2:
; LARGE-NEXT:  movk
; LARGE:  .Lpcsection3:
; LARGE-NEXT:  movk
; LARGE:  .Lpcsection4:
; --
; CHECK-NEXT:    ldr
; CHECK-NEXT:    ret
; CHECK:       .section	section_no_aux,"awo",@progbits,.text
; --
; DEFCM-NEXT:  .Lpcsection_base0:
; DEFCM-NEXT:  .word	.Lfunc_begin0-.Lpcsection_base0
; DEFCM-NEXT:  .word	.Lfunc_end0-.Lfunc_begin0
; DEFCM-NEXT:  .section	section_aux_42,"awo",@progbits,.text
; DEFCM-NEXT:  .Lpcsection_base1:
; DEFCM-NEXT:  .word	.Lpcsection0-.Lpcsection_base1
; DEFCM-NEXT:  .word	42
; DEFCM-NEXT:  .section	section_aux_21264,"awo",@progbits,.text
; DEFCM-NEXT:  .Lpcsection_base2:
; DEFCM-NEXT:  .word	.Lpcsection0-.Lpcsection_base2
; DEFCM-NEXT:  .word	21264
; --
; LARGE-NEXT: .Lpcsection_base0:
; LARGE-NEXT: 	.xword	.Lfunc_begin0-.Lpcsection_base0
; LARGE-NEXT: 	.word	.Lfunc_end0-.Lfunc_begin0
; LARGE-NEXT: 	.section	section_aux_42,"awo",@progbits,.text
; LARGE-NEXT: .Lpcsection_base1:
; LARGE-NEXT: 	.xword	.Lpcsection0-.Lpcsection_base1
; LARGE-NEXT: .Lpcsection_base2:
; LARGE-NEXT: 	.xword	.Lpcsection1-.Lpcsection_base2
; LARGE-NEXT: .Lpcsection_base3:
; LARGE-NEXT: 	.xword	.Lpcsection2-.Lpcsection_base3
; LARGE-NEXT: .Lpcsection_base4:
; LARGE-NEXT: 	.xword	.Lpcsection3-.Lpcsection_base4
; LARGE-NEXT: .Lpcsection_base5:
; LARGE-NEXT: 	.xword	.Lpcsection4-.Lpcsection_base5
; LARGE-NEXT: 	.word	42
; LARGE-NEXT: 	.section	section_aux_21264,"awo",@progbits,.text
; LARGE-NEXT: .Lpcsection_base6:
; LARGE-NEXT: 	.xword	.Lpcsection0-.Lpcsection_base6
; LARGE-NEXT: .Lpcsection_base7:
; LARGE-NEXT: 	.xword	.Lpcsection1-.Lpcsection_base7
; LARGE-NEXT: .Lpcsection_base8:
; LARGE-NEXT: 	.xword	.Lpcsection2-.Lpcsection_base8
; LARGE-NEXT: .Lpcsection_base9:
; LARGE-NEXT: 	.xword	.Lpcsection3-.Lpcsection_base9
; LARGE-NEXT: .Lpcsection_base10:
; LARGE-NEXT: 	.xword	.Lpcsection4-.Lpcsection_base10
; LARGE-NEXT: 	.word	21264
; --
; CHECK-NEXT: 	.text
entry:
  %0 = load i64, ptr @bar, align 8, !pcsections !1
  ret i64 %0
}

define i64 @test_simple_atomic() {
; CHECK-LABEL: test_simple_atomic:
; --
; DEFCM:       .Lpcsection1:
; DEFCM-NEXT:    ldr
; DEFCM-NOT:   .Lpcsection2
; DEFCM:         ldr
; --
; LARGE:  .Lpcsection5:
; LARGE-NEXT:  	movz
; LARGE-NEXT:  	movz
; LARGE:  .Lpcsection6:
; LARGE-NEXT:  	movk
; LARGE-NEXT:  	movk
; LARGE:  .Lpcsection7:
; LARGE-NEXT:  	movk
; LARGE-NEXT:  	movk
; LARGE:  .Lpcsection8:
; LARGE-NEXT:  	movk
; LARGE-NEXT:  	movk
; LARGE:  .Lpcsection9:
; LARGE-NEXT:  	ldr
; LARGE-NEXT:  	ldr
; --
; CHECK:         add
; CHECK-NEXT:    ret
; CHECK:       .section	section_no_aux,"awo",@progbits,.text
; --
; DEFCM-NEXT:  .Lpcsection_base3:
; DEFCM-NEXT:  .word	.Lpcsection1-.Lpcsection_base3
; --
; LARGE-NEXT:  .Lpcsection_base11:
; LARGE-NEXT:  .xword	.Lpcsection5-.Lpcsection_base11
; LARGE-NEXT:  .Lpcsection_base12:
; LARGE-NEXT:  .xword	.Lpcsection6-.Lpcsection_base12
; LARGE-NEXT:  .Lpcsection_base13:
; LARGE-NEXT:  .xword	.Lpcsection7-.Lpcsection_base13
; LARGE-NEXT:  .Lpcsection_base14:
; LARGE-NEXT:  .xword	.Lpcsection8-.Lpcsection_base14
; LARGE-NEXT:  .Lpcsection_base15:
; LARGE-NEXT:  .xword	.Lpcsection9-.Lpcsection_base15
; --
; CHECK-NEXT:  .text
entry:
  %0 = load atomic i64, ptr @foo monotonic, align 8, !pcsections !0
  %1 = load i64, ptr @bar, align 8
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
; LARGE:      .Lpcsection10:
; LARGE-NEXT: 	movz
; LARGE-NEXT: .Lpcsection11:
; LARGE-NEXT: 	movk
; LARGE-NEXT: .Lpcsection12:
; LARGE-NEXT: 	movk
; LARGE-NEXT: .Lpcsection13:
; LARGE-NEXT: 	movk
; LARGE:      .Lpcsection14:
; LARGE-NEXT: 	ldxr
; LARGE-NEXT: .Lpcsection15:
; LARGE-NEXT: 	add
; LARGE-NEXT: .Lpcsection16:
; LARGE-NEXT: 	stxr
; LARGE-NEXT: .Lpcsection17:
; LARGE-NEXT: 	cbnz
; ---
; CHECK-NOT:   .Lpcsection
; CHECK:         ldr
; CHECK:         ret
; CHECK:       .section	section_no_aux,"awo",@progbits,.text
; ---
; CHECK-OPT-NEXT: .Lpcsection_base4:
; CHECK-OPT-NEXT: 	.word	.Lpcsection2-.Lpcsection_base4
; CHECK-OPT-NEXT: .Lpcsection_base5:
; CHECK-OPT-NEXT: 	.word	.Lpcsection3-.Lpcsection_base5
; CHECK-OPT-NEXT: .Lpcsection_base6:
; CHECK-OPT-NEXT: 	.word	.Lpcsection4-.Lpcsection_base6
; CHECK-OPT-NEXT: .Lpcsection_base7:
; CHECK-OPT-NEXT: 	.word	.Lpcsection5-.Lpcsection_base7
; ---
; CHECK-UNOPT-NEXT: .Lpcsection_base4:
; CHECK-UNOPT-NEXT: 	.word	.Lpcsection2-.Lpcsection_base4
; CHECK-UNOPT-NEXT: .Lpcsection_base5:
; CHECK-UNOPT-NEXT: 	.word	.Lpcsection3-.Lpcsection_base5
; CHECK-UNOPT-NEXT: .Lpcsection_base6:
; CHECK-UNOPT-NEXT: 	.word	.Lpcsection4-.Lpcsection_base6
; CHECK-UNOPT-NEXT: .Lpcsection_base7:
; CHECK-UNOPT-NEXT: 	.word	.Lpcsection5-.Lpcsection_base7
; CHECK-UNOPT-NEXT: .Lpcsection_base8:
; CHECK-UNOPT-NEXT: 	.word	.Lpcsection6-.Lpcsection_base8
; CHECK-UNOPT-NEXT: .Lpcsection_base9:
; CHECK-UNOPT-NEXT: 	.word	.Lpcsection7-.Lpcsection_base9
; CHECK-UNOPT-NEXT: .Lpcsection_base10:
; CHECK-UNOPT-NEXT: 	.word	.Lpcsection8-.Lpcsection_base10
; CHECK-UNOPT-NEXT: .Lpcsection_base11:
; CHECK-UNOPT-NEXT: 	.word	.Lpcsection9-.Lpcsection_base11
; CHECK-UNOPT-NEXT: .Lpcsection_base12:
; CHECK-UNOPT-NEXT: 	.word	.Lpcsection10-.Lpcsection_base12
; CHECK-UNOPT-NEXT: .Lpcsection_base13:
; CHECK-UNOPT-NEXT: 	.word	.Lpcsection11-.Lpcsection_base13
; CHECK-UNOPT-NEXT: .Lpcsection_base14:
; CHECK-UNOPT-NEXT: 	.word	.Lpcsection12-.Lpcsection_base14
; CHECK-UNOPT-NEXT: .Lpcsection_base15:
; CHECK-UNOPT-NEXT: 	.word	.Lpcsection13-.Lpcsection_base15
; ---
; LARGE-NEXT: .Lpcsection_base16:
; LARGE-NEXT: 	.xword	.Lpcsection10-.Lpcsection_base16
; LARGE-NEXT: .Lpcsection_base17:
; LARGE-NEXT: 	.xword	.Lpcsection11-.Lpcsection_base17
; LARGE-NEXT: .Lpcsection_base18:
; LARGE-NEXT: 	.xword	.Lpcsection12-.Lpcsection_base18
; LARGE-NEXT: .Lpcsection_base19:
; LARGE-NEXT: 	.xword	.Lpcsection13-.Lpcsection_base19
; LARGE-NEXT: .Lpcsection_base20:
; LARGE-NEXT: 	.xword	.Lpcsection14-.Lpcsection_base20
; LARGE-NEXT: .Lpcsection_base21:
; LARGE-NEXT: 	.xword	.Lpcsection15-.Lpcsection_base21
; LARGE-NEXT: .Lpcsection_base22:
; LARGE-NEXT: 	.xword	.Lpcsection16-.Lpcsection_base22
; LARGE-NEXT: .Lpcsection_base23:
; LARGE-NEXT: 	.xword	.Lpcsection17-.Lpcsection_base23
; CHECK-NEXT:  .text
entry:
  %0 = atomicrmw add ptr @foo, i64 1 monotonic, align 8, !pcsections !0
  %1 = load i64, ptr @bar, align 8
  %inc = add nsw i64 %1, 1
  store i64 %inc, ptr @bar, align 8
  %add = add nsw i64 %1, %0
  ret i64 %add
}

!0 = !{!"section_no_aux"}
!1 = !{!"section_aux_42", !2, !"section_aux_21264", !3}
!2 = !{i32 42}
!3 = !{i32 21264}
