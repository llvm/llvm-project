; RUN: opt -objc-arc-contract -S < %s | FileCheck %s

; CHECK: tail call void @llvm.objc.storeStrong(i8**

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin11.0.0"

%0 = type opaque
%1 = type opaque

@"OBJC_IVAR_$_Controller.preferencesController" = external global i64, section "__DATA, __objc_const", align 8

declare i8* @llvm.objc.retain(i8*)

declare void @llvm.objc.release(i8*)

define hidden void @y(%0* nocapture %self, %1* %preferencesController) nounwind {
entry:
  %ivar = load i64, i64* @"OBJC_IVAR_$_Controller.preferencesController", align 8
  %tmp = bitcast %0* %self to i8*
  %add.ptr = getelementptr inbounds i8, i8* %tmp, i64 %ivar
  %tmp1 = bitcast i8* %add.ptr to %1**
  %tmp2 = load %1*, %1** %tmp1, align 8
  %tmp3 = bitcast %1* %preferencesController to i8*
  %tmp4 = tail call i8* @llvm.objc.retain(i8* %tmp3) nounwind
  %tmp5 = bitcast %1* %tmp2 to i8*
  tail call void @llvm.objc.release(i8* %tmp5) nounwind
  %tmp6 = bitcast i8* %tmp4 to %1*
  store %1* %tmp6, %1** %tmp1, align 8
  ret void
}
