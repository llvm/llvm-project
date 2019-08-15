; RUN: llc --frame-pointer=all -mtriple=aarch64-- < %s | FileCheck %s

; PR25610: -fstack-protector places the canary in the wrong place on arm64 with
;          va_args

%struct.__va_list = type { i8*, i8*, i8*, i32, i32 }

; CHECK-LABEL: test
; CHECK: ldr [[GUARD:x[0-9]+]]{{.*}}:lo12:__stack_chk_guard]
; Make sure the canary is placed relative to the frame pointer, not
; the stack pointer.
; CHECK: stur [[GUARD]], [x29, #-8]
define void @test(i8* %i, ...) #0 {
entry:
  %buf = alloca [10 x i8], align 1
  %ap = alloca %struct.__va_list, align 8
  %tmp = alloca %struct.__va_list, align 8
  %0 = getelementptr inbounds [10 x i8], [10 x i8]* %buf, i64 0, i64 0
  call void @llvm.lifetime.start(i64 10, i8* %0)
  %1 = bitcast %struct.__va_list* %ap to i8*
  call void @llvm.lifetime.start(i64 32, i8* %1)
  call void @llvm.va_start(i8* %1)
  %2 = bitcast %struct.__va_list* %tmp to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %2, i8* %1, i64 32, i32 8, i1 false)
  call void @baz(i8* %i, %struct.__va_list* nonnull %tmp)
  call void @bar(i8* %0)
  call void @llvm.va_end(i8* %1)
  call void @llvm.lifetime.end(i64 32, i8* %1)
  call void @llvm.lifetime.end(i64 10, i8* %0)
  ret void
}

declare void @llvm.lifetime.start(i64, i8* nocapture)
declare void @llvm.va_start(i8*)
declare void @baz(i8*, %struct.__va_list*)
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i32, i1)
declare void @bar(i8*)
declare void @llvm.va_end(i8*)
declare void @llvm.lifetime.end(i64, i8* nocapture)

attributes #0 = { noinline nounwind optnone ssp }
