; X86 GlobalISel can't lower inline asm, so only SelectionDAG is tested here;
; see the AArch64 test for both instruction selectors.
; RUN: not llc -mtriple=x86_64-unknown-linux-gnu -global-isel=0 \
; RUN:   -filetype=null < %s 2>&1 | FileCheck --implicit-check-not=error: %s

; A direct output is the asm's result, so there is no memory to write it to.
; Picking a memory constraint for one is an error, not a crash.

; "rm" picks memory, the most general constraint.
; CHECK: error: cannot handle direct memory outputs for constraint 'm' at line 1
define i32 @rm_i32() {
  %r = call i32 asm "# $0", "=rm"(), !srcloc !1
  ret i32 %r
}

; The same for a value no 'r' register can hold.
; CHECK: error: cannot handle direct memory outputs for constraint 'm' at line 2
define x86_fp80 @rm_x86_fp80() {
  %r = call x86_fp80 asm "# $0", "=rm"(), !srcloc !2
  ret x86_fp80 %r
}

; "=m" has no other choice.
; CHECK: error: cannot handle direct memory outputs for constraint 'm' at line 3
define i32 @m_i32() {
  %r = call i32 asm "# $0", "=m"(), !srcloc !3
  ret i32 %r
}

; With multiple alternatives, the memory one is chosen here, too.
; CHECK: error: cannot handle direct memory outputs for constraint 'm' at line 4
define i32 @multi_alt() {
  %r = call i32 asm "# $0", "=m|r"(), !srcloc !4
  ret i32 %r
}

; A label is a branch target, which isn't in memory either. It isn't an
; output, so the error says so.
; CHECK: error: cannot handle memory constraint 'm' for a label at line 5
define void @label_mem() {
  callbr void asm "# $0", "!m"()
    to label %fallthrough [label %indirect], !srcloc !5
fallthrough:
  ret void
indirect:
  ret void
}

; The same for an indirect label.
; CHECK: error: cannot handle memory constraint 'm' for a label at line 6
define void @label_indirect_mem() {
  callbr void asm "# $0", "!*m"()
    to label %fallthrough [label %indirect], !srcloc !6
fallthrough:
  ret void
indirect:
  ret void
}

; Nor does a label have a value to take the address of or to point to, so it
; can't take an address constraint ("p") or be indirect.
; CHECK: error: cannot handle address constraint 'p' for a label at line 7
define void @test_label_address() {
  callbr void asm "# $0", "!p"()
    to label %fallthrough [label %indirect], !srcloc !7
fallthrough:
  ret void
indirect:
  ret void
}

; CHECK: error: cannot handle address constraint 'p' for a label at line 8
define void @test_label_indirect_address() {
  callbr void asm "# $0", "!*p"()
    to label %fallthrough [label %indirect], !srcloc !8
fallthrough:
  ret void
indirect:
  ret void
}

; CHECK: error: cannot handle indirect labels for constraint 'i' at line 9
define void @test_label_indirect_imm() {
  callbr void asm "# $0", "!*i"()
    to label %fallthrough [label %indirect], !srcloc !9
fallthrough:
  ret void
indirect:
  ret void
}

; "X" picks "i" for a label, which the error names.
; CHECK: error: cannot handle indirect labels for constraint 'i' at line 10
define void @test_label_indirect_any() {
  callbr void asm "# $0", "!*X"()
    to label %fallthrough [label %indirect], !srcloc !10
fallthrough:
  ret void
indirect:
  ret void
}

; An address constraint passes an address to the asm, which only an input has.
; CHECK: error: cannot handle address constraint 'p' for an output at line 11
define i64 @test_address_output() {
  %r = call i64 asm "# $0", "=p"(), !srcloc !11
  ret i64 %r
}

; CHECK: error: cannot handle address constraint 'p' for an output at line 12
define void @test_indirect_address_output(ptr %p) {
  call void asm "# $0", "=*p"(ptr elementtype(i64) %p), !srcloc !12
  ret void
}

!1 = !{i64 1}
!2 = !{i64 2}
!3 = !{i64 3}
!4 = !{i64 4}
!5 = !{i64 5}
!6 = !{i64 6}
!7 = !{i64 7}
!8 = !{i64 8}
!9 = !{i64 9}
!10 = !{i64 10}
!11 = !{i64 11}
!12 = !{i64 12}
