; X86 GlobalISel can't lower inline asm, so only SelectionDAG is tested here;
; see the AArch64 test for both instruction selectors. The two RUN lines differ
; in optimization level because "rm" prefers a register only above -O0.
; RUN: not llc -mtriple=x86_64-unknown-linux-gnu -global-isel=0 -O0 \
; RUN:   -filetype=null < %s 2>&1 \
; RUN:   | FileCheck --check-prefixes=CHECK,O0 --implicit-check-not=error: %s
; RUN: not llc -mtriple=x86_64-unknown-linux-gnu -global-isel=0 \
; RUN:   -filetype=null < %s 2>&1 | FileCheck --implicit-check-not=error: %s

; A direct output is the asm's result, so there is no memory to write it to.
; Picking a memory constraint for one is an error, not a crash.

; -O0 picks memory for "rm"; above -O0 the register is preferred.
; O0: error: cannot handle direct memory outputs for constraint 'm'
define i32 @rm_i32() {
  %r = call i32 asm "# $0", "=rm"()
  ret i32 %r
}

; No 'r' register can hold an x87 value, so "rm" picks memory either way.
; CHECK: error: cannot handle direct memory outputs for constraint 'm'
define x86_fp80 @rm_x86_fp80() {
  %r = call x86_fp80 asm "# $0", "=rm"()
  ret x86_fp80 %r
}

; "=m" has no other choice.
; CHECK: error: cannot handle direct memory outputs for constraint 'm'
define i32 @m_i32() {
  %r = call i32 asm "# $0", "=m"()
  ret i32 %r
}

; With multiple alternatives, the memory one is chosen here, too.
; CHECK: error: cannot handle direct memory outputs for constraint 'm'
define i32 @multi_alt() {
  %r = call i32 asm "# $0", "=m|r"()
  ret i32 %r
}
