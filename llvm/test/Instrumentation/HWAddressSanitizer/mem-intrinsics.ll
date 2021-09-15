; RUN: opt -S -passes=hwasan -hwasan-use-stack-safety=0 %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() sanitize_hwaddress {
entry:
  %retval = alloca i32, align 4
  %Q = alloca [10 x i8], align 1
  %P = alloca [10 x i8], align 1
  store i32 0, i32* %retval, align 4
  %arraydecay = getelementptr inbounds [10 x i8], [10 x i8]* %Q, i32 0, i32 0

  call void @llvm.memset.p0i8.i64(i8* align 1 %arraydecay, i8 0, i64 10, i1 false)
; CHECK: call i8* @__hwasan_memset

  %arraydecay1 = getelementptr inbounds [10 x i8], [10 x i8]* %Q, i32 0, i32 0
  %arraydecay2 = getelementptr inbounds [10 x i8], [10 x i8]* %Q, i32 0, i32 0
  %add.ptr = getelementptr inbounds i8, i8* %arraydecay2, i64 5

  call void @llvm.memmove.p0i8.p0i8.i64(i8* align 1 %arraydecay1, i8* align 1 %add.ptr, i64 5, i1 false)
; CHECK: call i8* @__hwasan_memmove

  %arraydecay3 = getelementptr inbounds [10 x i8], [10 x i8]* %P, i32 0, i32 0
  %arraydecay4 = getelementptr inbounds [10 x i8], [10 x i8]* %Q, i32 0, i32 0

  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %arraydecay3, i8* align 1 %arraydecay4, i64 10, i1 false)
; CHECK: call i8* @__hwasan_memcpy
  ret i32 0
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.memmove.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i1) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1) #1
