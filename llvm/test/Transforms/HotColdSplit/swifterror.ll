; RUN: opt -passes=hotcoldsplit -hotcoldsplit-threshold=0 -S < %s | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.14.0"

%swift_error = type {i64, i8}

declare void @sink() cold

; CHECK-LABEL: define {{.*}}@in_arg(
; CHECK: call void @in_arg.cold.1(ptr swifterror
define void @in_arg(ptr swifterror %error_ptr_ref) {
  br i1 undef, label %cold, label %exit

cold:
  store ptr undef, ptr %error_ptr_ref
  call void @sink()
  br label %exit

exit:
  ret void
}

; CHECK-LABEL: define {{.*}}@in_alloca(
; CHECK: call void @in_alloca.cold.1(ptr swifterror
define void @in_alloca() {
  %err = alloca swifterror ptr
  br i1 undef, label %cold, label %exit

cold:
  store ptr undef, ptr %err
  call void @sink()
  br label %exit

exit:
  ret void
}

; CHECK-LABEL: define {{.*}}@in_arg.cold.1({{.*}} swifterror
; CHECK: call {{.*}}@sink

; CHECK-LABEL: define {{.*}}@in_alloca.cold.1({{.*}} swifterror
; CHECK: call {{.*}}@sink
