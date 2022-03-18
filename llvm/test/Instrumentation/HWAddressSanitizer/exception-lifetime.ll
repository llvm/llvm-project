; Test allocas with multiple lifetime ends, as frequently seen for exception
; handling.
;
; RUN: opt -passes=hwasan -hwasan-use-after-scope -S -o - %s | FileCheck %s --check-prefix=CHECK

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-android"

declare void @mayFail(i32* %x) sanitize_hwaddress
declare void @onExcept(i32* %x) sanitize_hwaddress

declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) nounwind
declare i32 @__gxx_personality_v0(...)

define void @test() sanitize_hwaddress personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %x = alloca i32, align 8
  %exn.slot = alloca i8*, align 8
  %ehselector.slot = alloca i32, align 4
  %0 = bitcast i32* %x to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %0)
  invoke void @mayFail(i32* %x) to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %entry
; CHECK: invoke.cont:
; CHECK:  call void @llvm.memset.p0i8.i64(i8* align 1 %31, i8 0, i64 1, i1 false)
; CHECK:  call void @llvm.lifetime.end.p0i8(i64 8, i8* %28)
; CHECK:  ret void

  %1 = bitcast i32* %x to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %1)
  ret void

lpad:                                             ; preds = %entry
; CHECK: lpad
; CHECK:  %41 = getelementptr i8, i8* %17, i64 %40
; CHECK:  call void @llvm.memset.p0i8.i64(i8* align 1 %41, i8 0, i64 1, i1 false)
; CHECK:  call void @llvm.lifetime.end.p0i8(i64 8, i8* %38)
; CHECK:  br label %eh.resume

  %2 = landingpad { i8*, i32 }
  cleanup
  %3 = extractvalue { i8*, i32 } %2, 0
  store i8* %3, i8** %exn.slot, align 8
  %4 = extractvalue { i8*, i32 } %2, 1
  store i32 %4, i32* %ehselector.slot, align 4
  call void @onExcept(i32* %x) #18
  %5 = bitcast i32* %x to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %5)
  br label %eh.resume

eh.resume:                                        ; preds = %lpad
  %exn = load i8*, i8** %exn.slot, align 8
  %sel = load i32, i32* %ehselector.slot, align 4
  %lpad.val = insertvalue { i8*, i32 } undef, i8* %exn, 0
  %lpad.val1 = insertvalue { i8*, i32 } %lpad.val, i32 %sel, 1
  resume { i8*, i32 } %lpad.val1
}
