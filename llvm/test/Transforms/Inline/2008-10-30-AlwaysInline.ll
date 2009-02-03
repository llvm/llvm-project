; RUN: llvm-as < %s | opt -always-inline | llvm-dis | not grep call 

; Ensure that threshold doesn't disrupt always inline.
; RUN: llvm-as < %s | opt -inline-threshold=-2000000001 -always-inline | llvm-dis | not grep call 


define internal i32 @if0() alwaysinline {
       ret i32 1 
}

define i32 @f0() {
       %r = call i32 @if0()
       ret i32 %r
}
