; RUN: opt -hotcoldsplit -hotcoldsplit-threshold=0 -S < %s | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.14.0"

%struct.__jmp_buf_tag = type { [8 x i64], i32, %struct.__sigset_t }
%struct.__sigset_t = type { [16 x i64] }

; Don't outline noreturn calls which aren't explicitly marked cold.

; CHECK-LABEL: define {{.*}}@foo(
; CHECK-NOT: foo.cold.1
define void @foo(i32, ptr) {
  %3 = icmp eq i32 %0, 0
  tail call void @_Z10sideeffectv()
  br i1 %3, label %5, label %4

; <label>:4:                                      ; preds = %2
  tail call void @longjmp(ptr %1, i32 0)
  unreachable

; <label>:5:                                      ; preds = %2
  ret void
}

; Don't outline within a noreturn function.

; CHECK: define {{.*}}@xpc_objc_main(i32 {{.*}}) [[XPC_OBJC_MAIN_ATTRS:#[0-9]+]]
; CHECK-NOT: xpc_objc_main.cold.1
define void @xpc_objc_main(i32) noreturn {
  %2 = icmp eq i32 %0, 0
  tail call void @_Z10sideeffectv()
  br i1 %2, label %4, label %3

; <label>:3:                                      ; preds = %2
  call void @_Z10sideeffectv()
  unreachable

; <label>:4:                                      ; preds = %2
  ; Crash with an error message, "not supposed to return".
  unreachable
}

; Do outline noreturn calls marked cold.

; CHECK-LABEL: define {{.*}}@bar(
; CHECK: call {{.*}}@bar.cold.1(
define void @bar(i32) {
  %2 = icmp eq i32 %0, 0
  tail call void @_Z10sideeffectv()
  br i1 %2, label %sink, label %exit

sink:
  tail call void @_Z10sideeffectv()
  call void @llvm.trap()
  unreachable

exit:
  ret void
}

; Do outline noreturn calls preceded by a cold call.

; CHECK-LABEL: define {{.*}}@baz(
; CHECK: call {{.*}}@baz.cold.1(
define void @baz(i32, ptr) {
  %3 = icmp eq i32 %0, 0
  tail call void @_Z10sideeffectv()
  br i1 %3, label %5, label %4

; <label>:4:                                      ; preds = %2
  call void @sink()
  tail call void @longjmp(ptr %1, i32 0)
  unreachable

; <label>:5:                                      ; preds = %2
  ret void
}

; CHECK-LABEL: define {{.*}}@bar.cold.1(
; CHECK: call {{.*}}@llvm.trap(

; CHECK: attributes [[XPC_OBJC_MAIN_ATTRS]] = { noreturn }

declare void @sink() cold

declare void @llvm.trap() noreturn cold

declare void @_Z10sideeffectv()

declare void @longjmp(ptr, i32) noreturn nounwind
