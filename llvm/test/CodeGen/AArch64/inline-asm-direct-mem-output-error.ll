; RUN: not llc -mtriple=aarch64 -global-isel=0 -filetype=null < %s 2>&1 \
; RUN:   | FileCheck --implicit-check-not=error: %s
; RUN: not llc -mtriple=aarch64 -global-isel -global-isel-abort=1 \
; RUN:   -filetype=null < %s 2>&1 | FileCheck --implicit-check-not=error: %s

; A direct output is the asm's result, so there is no memory to write it to.
; Picking a memory constraint for one is an error, not a crash, with either
; instruction selector. AArch64 can't fold a register operand to memory, so
; "rm" keeps picking memory.

; CHECK: error: cannot handle direct memory outputs for constraint 'm'
define i64 @rm_output() {
  %r = call i64 asm "# $0", "=rm"()
  ret i64 %r
}

; With multiple alternatives, the memory one is chosen here, too.
; CHECK: error: cannot handle direct memory outputs for constraint 'm'
define i64 @multi_alt() {
  %r = call i64 asm "# $0", "=m|r"()
  ret i64 %r
}
