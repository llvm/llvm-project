; RUN: opt < %s -analyze -inline -enable-new-pm=0
; PR1526
; RUN: opt < %s -analyze -indvars -enable-new-pm=0
; PR1539
define i32 @test1() {
       ret i32 0
}
