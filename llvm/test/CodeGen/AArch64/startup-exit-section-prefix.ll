; REQUIRES: aarch64-registered-target

; RUN: llc -mtriple aarch64-linux-gnu -o - %s | FileCheck %s --check-prefix=PLAIN
; RUN: llc -mtriple aarch64-linux-gnu -function-sections -o - %s | FileCheck %s --check-prefix=FUNC-SECT

;; A constructor function receives section_prefix "startup".
; PLAIN:      .section .text.startup.,"ax",@progbits
; FUNC-SECT:  .section .text.startup.my_ctor,"ax",@progbits
define void @my_ctor() !section_prefix !0 {
  ret void
}

;; A destructor function receives section_prefix "exit".
; PLAIN:      .section .text.exit.,"ax",@progbits
; FUNC-SECT:  .section .text.exit.my_dtor,"ax",@progbits
define void @my_dtor() !section_prefix !1 {
  ret void
}

;; A regular function literally named "exit". With -function-sections it
;; gets .text.exit (derived from the function name, no trailing dot);
;; without -function-sections it stays in the generic .text section.
; PLAIN-NOT:  .section .text.exit.
; FUNC-SECT:  .section .text.exit,"ax",@progbits
define void @exit() {
  ret void
}

!0 = !{!"section_prefix", !"startup"}
!1 = !{!"section_prefix", !"exit"}
