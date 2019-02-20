; RUN: opt < %s -ipsccp -S | FileCheck %s
; PR36485
; musttail call result can\'t be replaced with a constant, unless the call
; can be removed

declare i32 @external()

define i8* @start(i8 %v) {
  %c1 = icmp eq i8 %v, 0
  br i1 %c1, label %true, label %false
true:
  ; CHECK: %ca = musttail call i8* @side_effects(i8 0)
  ; CHECK: ret i8* %ca
  %ca = musttail call i8* @side_effects(i8 %v)
  ret i8* %ca
false:
  %c2 = icmp eq i8 %v, 1
  br i1 %c2, label %c2_true, label %c2_false
c2_true:
  %ca1 = musttail call i8* @no_side_effects(i8 %v)
  ; CHECK: ret i8* null
  ret i8* %ca1
c2_false:
  ; CHECK: %ca2 = musttail call i8* @dont_zap_me(i8 %v)
  ; CHECK: ret i8* %ca2
  %ca2 = musttail call i8* @dont_zap_me(i8 %v)
  ret i8* %ca2
}

define internal i8* @side_effects(i8 %v) {
  %i1 = call i32 @external()

  ; since this goes back to `start` the SCPP should be see that the return value
  ; is always `null`.
  ; The call can't be removed due to `external` call above, though.

  ; CHECK: %ca = musttail call i8* @start(i8 0)
  %ca = musttail call i8* @start(i8 %v)

  ; Thus the result must be returned anyway
  ; CHECK: ret i8* %ca
  ret i8* %ca
}

define internal i8* @no_side_effects(i8 %v) readonly nounwind {
  ; The call to this function is removed, so the return value must be zapped
  ; CHECK: ret i8* undef
  ret i8* null
}

define internal i8* @dont_zap_me(i8 %v) {
  %i1 = call i32 @external()

  ; The call to this function cannot be removed due to side effects. Thus the
  ; return value should stay as it is, and should not be zapped.
  ; CHECK: ret i8* null
  ret i8* null
}
