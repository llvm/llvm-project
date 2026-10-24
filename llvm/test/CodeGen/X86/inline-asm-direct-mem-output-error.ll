; X86 GlobalISel can't lower inline asm, so only SelectionDAG is tested here;
; see the AArch64 test for both instruction selectors.
; RUN: not llc -mtriple=x86_64-unknown-linux-gnu -global-isel=0 \
; RUN:   -filetype=null < %s 2>&1 | FileCheck --implicit-check-not=error: %s

; A direct output is the asm's result, so there is no memory to write it to.
; Picking a memory constraint for one is an error, not a crash.

; "rm" picks memory, the most general constraint.
; CHECK: error: cannot handle direct memory outputs for constraint 'm'
define i32 @rm_i32() {
  %r = call i32 asm "# $0", "=rm"()
  ret i32 %r
}

; The same for a value no 'r' register can hold.
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

; A label is a branch target, which isn't in memory either. It isn't an
; output, so the error says so.
; CHECK: error: cannot handle memory constraint 'm' for a label
define void @label_mem() {
  callbr void asm "# $0", "!m"()
    to label %fallthrough [label %indirect]
fallthrough:
  ret void
indirect:
  ret void
}
