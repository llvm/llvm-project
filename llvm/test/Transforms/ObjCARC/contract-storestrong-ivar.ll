; RUN: opt -passes=objc-arc-contract -S < %s | FileCheck %s

; CHECK: tail call void @llvm.objc.storeStrong(ptr

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin11.0.0"

%0 = type opaque
%1 = type opaque

@"OBJC_IVAR_$_Controller.preferencesController" = external global i64, section "__DATA, __objc_const", align 8

declare ptr @llvm.objc.retain(ptr)

declare void @llvm.objc.release(ptr)

define hidden void @y(ptr nocapture %self, ptr %preferencesController) nounwind {
entry:
  %ivar = load i64, ptr @"OBJC_IVAR_$_Controller.preferencesController", align 8
  %add.ptr = getelementptr inbounds i8, ptr %self, i64 %ivar
  %tmp2 = load ptr, ptr %add.ptr, align 8
  %tmp4 = tail call ptr @llvm.objc.retain(ptr %preferencesController) nounwind
  tail call void @llvm.objc.release(ptr %tmp2) nounwind
  store ptr %tmp4, ptr %add.ptr, align 8
  ret void
}
