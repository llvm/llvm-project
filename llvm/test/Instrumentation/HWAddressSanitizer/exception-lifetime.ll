; Test allocas with multiple lifetime ends, as frequently seen for exception
; handling.
;
; RUN: opt -passes=hwasan -hwasan-use-after-scope -S -o - %s | FileCheck %s --check-prefix=CHECK

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-android"

declare void @mayFail(ptr %x) sanitize_hwaddress
declare void @onExcept(ptr %x) sanitize_hwaddress

declare void @llvm.lifetime.start.p0(i64, ptr nocapture) nounwind
declare void @llvm.lifetime.end.p0(i64, ptr nocapture) nounwind
declare i32 @__gxx_personality_v0(...)

define void @test() sanitize_hwaddress personality ptr @__gxx_personality_v0 {
entry:
  %x = alloca i32, align 8
  %exn.slot = alloca ptr, align 8
  %ehselector.slot = alloca i32, align 4
  call void @llvm.lifetime.start.p0(i64 8, ptr %x)
  invoke void @mayFail(ptr %x) to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %entry
; CHECK: invoke.cont:
; CHECK:  call void @llvm.memset.p0.i64(ptr align 1 %{{.*}}, i8 0, i64 1, i1 false)
; CHECK:  call void @llvm.lifetime.end.p0(i64 16, ptr {{.*}}{{.*}}%x)
; CHECK:  ret void

  call void @llvm.lifetime.end.p0(i64 8, ptr %x)
  ret void

lpad:                                             ; preds = %entry
; CHECK: lpad
; CHECK:  call void @llvm.memset.p0.i64(ptr align 1 %{{.*}}, i8 0, i64 1, i1 false)
; CHECK:  call void @llvm.lifetime.end.p0(i64 16, ptr {{.*}}{{.*}}%x)
; CHECK:  br label %eh.resume

  %0 = landingpad { ptr, i32 }
  cleanup
  %1 = extractvalue { ptr, i32 } %0, 0
  store ptr %1, ptr %exn.slot, align 8
  %2 = extractvalue { ptr, i32 } %0, 1
  store i32 %2, ptr %ehselector.slot, align 4
  call void @onExcept(ptr %x) #18
  call void @llvm.lifetime.end.p0(i64 8, ptr %x)
  br label %eh.resume

eh.resume:                                        ; preds = %lpad
  %exn = load ptr, ptr %exn.slot, align 8
  %sel = load i32, ptr %ehselector.slot, align 4
  %lpad.val = insertvalue { ptr, i32 } undef, ptr %exn, 0
  %lpad.val1 = insertvalue { ptr, i32 } %lpad.val, i32 %sel, 1
  resume { ptr, i32 } %lpad.val1
}
