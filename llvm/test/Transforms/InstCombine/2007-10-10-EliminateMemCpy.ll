; RUN: opt < %s -passes=instcombine -S | not grep " call"
; RUN: opt < %s -O3 -S | not grep xyz
target datalayout = "E-p:64:64:64-a0:0:8-f32:32:32-f64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-v64:64:64-v128:128:128"

@.str = internal constant [4 x i8] c"xyz\00"		; <ptr> [#uses=1]

define void @foo(ptr %P) {
entry:
  %P_addr = alloca ptr
  store ptr %P, ptr %P_addr
  %tmp = load ptr, ptr %P_addr, align 4
  call void @llvm.memcpy.p0.p0.i32(ptr %tmp, ptr @.str, i32 4, i1 false)
  br label %return

return:                                           ; preds = %entry
  ret void
}

declare void @llvm.memcpy.p0.p0.i32(ptr nocapture, ptr nocapture, i32, i1) nounwind
