; RUN: opt < %s -mtriple=arm-arm-none-eabi -passes=globalopt -S | FileCheck %s
; RUN: opt < %s -mtriple=arm-arm-none-eabi -passes="default<O0>" -S | FileCheck %s --check-prefix=TURNED-OFF

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"

; CHECK: [12 x i8]
; TURNED-OFF-NOT: [12 x i8]
@.str = private unnamed_addr constant [10 x i8] c"123456789\00", align 1

define hidden void @foo() local_unnamed_addr {
entry:
; CHECK: %something = alloca [12 x i8]
; TURNED-OFF-NOT: %something = alloca [12 x i8]
  %something = alloca [10 x i8], align 1
  call void @llvm.memcpy.p0.p0.i32(ptr noundef nonnull align 1 dereferenceable(10) %something, ptr noundef nonnull align 1 dereferenceable(10) @.str, i32 10, i1 false)
  %call2 = call i32 @bar(ptr nonnull %something)
  ret void
}

declare i32 @bar(...) local_unnamed_addr

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i32(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i32, i1 immarg) #0

attributes #0 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
