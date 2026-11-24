; RUN: not llc -mtriple=aarch64 -global-isel=0 -filetype=null < %s 2>&1 \
; RUN:   | FileCheck --implicit-check-not=error: %s
; RUN: not llc -mtriple=aarch64 -global-isel -global-isel-abort=1 \
; RUN:   -filetype=null < %s 2>&1 | FileCheck --implicit-check-not=error: %s

; A direct output is the asm's result, so there is no memory to write it to.
; Picking a memory constraint for one is an error, not a crash, with either
; instruction selector.

; CHECK: error: cannot handle direct memory outputs for constraint 'm' at line 1
define i64 @rm_output() {
  %r = call i64 asm "# $0", "=rm"(), !srcloc !1
  ret i64 %r
}

; With multiple alternatives, the memory one is chosen here, too.
; CHECK: error: cannot handle direct memory outputs for constraint 'm' at line 2
define i64 @multi_alt() {
  %r = call i64 asm "# $0", "=m|r"(), !srcloc !2
  ret i64 %r
}

; An address constraint ("p") passes an address to the asm, which only an
; input has.
; CHECK: error: cannot handle address constraint 'p' for an output at line 3
define i64 @test_address_output() {
  %r = call i64 asm "# $0", "=p"(), !srcloc !3
  ret i64 %r
}

; CHECK: error: cannot handle address constraint 'p' for an output at line 4
define void @test_indirect_address_output(ptr %p) {
  call void asm "# $0", "=*p"(ptr elementtype(i64) %p), !srcloc !4
  ret void
}

!1 = !{i64 1}
!2 = !{i64 2}
!3 = !{i64 3}
!4 = !{i64 4}
