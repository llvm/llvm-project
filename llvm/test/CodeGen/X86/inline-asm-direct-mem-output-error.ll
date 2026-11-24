; X86 GlobalISel can't lower inline asm, so only SelectionDAG is tested here;
; see the AArch64 test for both instruction selectors.
; RUN: not llc -mtriple=x86_64-unknown-linux-gnu -global-isel=0 \
; RUN:   -filetype=null < %s 2>&1 | FileCheck --implicit-check-not=error: %s

; A direct output is the asm's result, so there is no memory to write it to.
; Picking a memory constraint for one is an error, not a crash.

; "rm" prefers the register here, so there's no error.
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

; The same for an indirect label.
; CHECK: error: cannot handle memory constraint 'm' for a label
define void @label_indirect_mem() {
  callbr void asm "# $0", "!*m"()
    to label %fallthrough [label %indirect]
fallthrough:
  ret void
indirect:
  ret void
}

; Nor does a label have a value to take the address of or to point to, so it
; can't take an address constraint ("p") or be indirect.
; CHECK: error: cannot handle address constraint 'p' for a label
define void @test_label_address() {
  callbr void asm "# $0", "!p"()
    to label %fallthrough [label %indirect]
fallthrough:
  ret void
indirect:
  ret void
}

; CHECK: error: cannot handle address constraint 'p' for a label
define void @test_label_indirect_address() {
  callbr void asm "# $0", "!*p"()
    to label %fallthrough [label %indirect]
fallthrough:
  ret void
indirect:
  ret void
}

; CHECK: error: cannot handle indirect labels for constraint 'i'
define void @test_label_indirect_imm() {
  callbr void asm "# $0", "!*i"()
    to label %fallthrough [label %indirect]
fallthrough:
  ret void
indirect:
  ret void
}

; "X" picks "i" for a label, which the error names.
; CHECK: error: cannot handle indirect labels for constraint 'i'
define void @test_label_indirect_any() {
  callbr void asm "# $0", "!*X"()
    to label %fallthrough [label %indirect]
fallthrough:
  ret void
indirect:
  ret void
}

; An address constraint passes an address to the asm, which only an input has.
; CHECK: error: cannot handle address constraint 'p' for an output
define i64 @test_address_output() {
  %r = call i64 asm "# $0", "=p"()
  ret i64 %r
}

; CHECK: error: cannot handle address constraint 'p' for an output
define void @test_indirect_address_output(ptr %p) {
  call void asm "# $0", "=*p"(ptr elementtype(i64) %p)
  ret void
}
