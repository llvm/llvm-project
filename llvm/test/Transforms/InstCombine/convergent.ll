; RUN: opt -passes=instcombine -S < %s | FileCheck -enable-var-scope %s

declare i32 @k() convergent
declare i32 @f()

declare i64 @llvm.read_register.i64(metadata) nounwind

define i32 @extern() {
  ; Convergent attr shouldn't be removed here; k is convergent.
  ; CHECK: call i32 @k() [[$CONVERGENT_ATTR:#[0-9]+]]
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

define i32 @indirect_call(ptr %f) {
  ; CHECK: call i32 %f() [[$CONVERGENT_ATTR]]
  %a = call i32 %f() convergent
  ret i32 %a
}

; do not remove from convergent intrinsic call sites
; CHECK-LABEL: @convergent_intrinsic_call(
; CHECK: call i64 @llvm.read_register.i64(metadata !0) [[$CONVERGENT_ATTR]]
define i64 @convergent_intrinsic_call() {
  %val = call i64 @llvm.read_register.i64(metadata !0) convergent
  ret i64 %val
}

; CHECK: [[$CONVERGENT_ATTR]] = { convergent }
!0 = !{!"foo"}
