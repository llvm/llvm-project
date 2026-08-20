; RUN: llc -O0 %s --basic-block-sections=all -mtriple=x86_64-- --frame-pointer=all -filetype=asm -o - | FileCheck %s
; RUN: llc -O0 %s --basic-block-sections=all -mtriple=x86_64-- --frame-pointer=all -filetype=obj -o %t.o
; RUN: llvm-dwarfdump --verify --eh-frame %t.o | FileCheck %s --check-prefix=VERIFY

;; AsmPrinter::emitCFIInstruction skips a trailing CFI (one with no following
;; "real" instruction) because such a directive would lie beyond the end of its
;; FDE. The block boundary that matters is the end of the current *section*, not
;; just the end of the whole function: with -basic-block-sections / function
;; splitting a single function is emitted as several FDEs (one per section).
;;
;; Here the two empty `unreachable` blocks %d0 and %d1 each become their own
;; section/FDE under -basic-block-sections=all, and neither is the function's
;; last block (%ret0 is). The trailing CFI in those non-final sections must
;; still be skipped (guarded by MBB->isEndSection()), so each empty section
;; comes out bare: just .cfi_startproc / .cfi_endproc, with no .cfi_def_cfa or
;; .cfi_offset in between.

; The function entry still emits its prologue CFI.
; CHECK-LABEL: trailing_cfi_section_end:
; CHECK:       .cfi_startproc
; CHECK:       .cfi_def_cfa_offset 16

; %d0 - non-final empty section: trailing CFI must be skipped.
; CHECK-LABEL: __part.{{[0-9]+}}: {{.*}}%d0
; CHECK-NEXT:  .cfi_startproc
; CHECK-NOT:   .cfi_def_cfa
; CHECK-NOT:   .cfi_offset
; CHECK:       .cfi_endproc

; %d1 - non-final empty section: trailing CFI must be skipped.
; CHECK-LABEL: __part.{{[0-9]+}}: {{.*}}%d1
; CHECK-NEXT:  .cfi_startproc
; CHECK-NOT:   .cfi_def_cfa
; CHECK-NOT:   .cfi_offset
; CHECK:       .cfi_endproc

; %ret0 - a real section keeps its CFI (we must not over-suppress).
; CHECK-LABEL: __part.{{[0-9]+}}: {{.*}}%ret0
; CHECK-NEXT:  .cfi_startproc
; CHECK:       .cfi_def_cfa
; CHECK:       .cfi_endproc

; The emitted unwind info must remain well-formed.
; VERIFY: No errors.

target triple = "x86_64-unknown-linux-gnu"

declare void @ext()

define void @trailing_cfi_section_end(i32 %x) #0 {
entry:
  switch i32 %x, label %ret0 [ i32 0, label %d0
                               i32 1, label %d1 ]
d0:
  unreachable
d1:
  unreachable
ret0:
  call void @ext()
  ret void
}

attributes #0 = { uwtable "frame-pointer"="all" }
