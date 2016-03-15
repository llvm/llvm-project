; RUN: opt -instcombine -S < %s | FileCheck %s

declare i32 @k() convergent
declare i32 @f()

define i32 @extern() {
  ; Convergent attr shouldn't be removed here; k is convergent.
  ; CHECK: call i32 @k() [[CONVERGENT_ATTR:#[0-9]+]]
  %a = call i32 @k() convergent
  ret i32 %a
}

define i32 @extern_no_attr() {
  ; Convergent attr shouldn't be added here, even though k is convergent.
  ; CHECK: call i32 @k(){{$}}
  %a = call i32 @k()
  ret i32 %a
}

define i32 @no_extern() {
  ; Convergent should be removed here, as the target is convergent.
  ; CHECK: call i32 @f(){{$}}
  %a = call i32 @f() convergent
  ret i32 %a
}

define i32 @indirect_call(i32 ()* %f) {
  ; CHECK call i32 %f() [[CONVERGENT_ATTR]]
  %a = call i32 %f() convergent
  ret i32 %a
}

; CHECK: [[CONVERGENT_ATTR]] = { convergent }
