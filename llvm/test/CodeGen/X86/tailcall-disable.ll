; RUN: llc -disable-tail-calls < %s | FileCheck --check-prefix=CALL %s
; RUN: llc < %s | FileCheck --check-prefix=JMP %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @helper() nounwind {
entry:
  ret i32 7
}

define i32 @test1() nounwind {
entry:
  %call = tail call i32 @helper()
  ret i32 %call
}

; CALL: test1:
; CALL-NOT: ret
; CALL: callq helper
; CALL: ret

; JMP: test1:
; JMP-NOT: ret
; JMP: jmp helper # TAILCALL

define i32 @test2() nounwind {
entry:
  %call = tail call i32 @test2()
  ret i32 %call
}

; CALL: test2:
; CALL-NOT: ret
; CALL: callq test2
; CALL: ret

; JMP: test2:
; JMP-NOT: ret
; JMP: jmp test2 # TAILCALL
