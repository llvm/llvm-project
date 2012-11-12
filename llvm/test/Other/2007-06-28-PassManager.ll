; RUN: opt < %s -analyze -inline
; PR1526
; RUN: opt < %s -analyze -indvars
; PR1539
define i32 @test1() {
       ret i32 0
}
