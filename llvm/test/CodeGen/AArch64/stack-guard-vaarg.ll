; RUN: llc --frame-pointer=all -mtriple=aarch64-- < %s | FileCheck %s

; PR25610: -fstack-protector places the canary in the wrong place on arm64 with
;          va_args

%struct.__va_list = type { ptr, ptr, ptr, i32, i32 }

; CHECK-LABEL: test
; CHECK: ldr [[GUARD:x[0-9]+]]{{.*}}:lo12:__stack_chk_guard]
; Make sure the canary is placed relative to the frame pointer, not
; the stack pointer.
; CHECK: stur [[GUARD]], [x29, #-8]
define void @test(ptr %i, ...) #0 {
entry:
  %buf = alloca [10 x i8], align 1
  %ap = alloca %struct.__va_list, align 8
  %tmp = alloca %struct.__va_list, align 8
  call void @llvm.lifetime.start(i64 10, ptr %buf)
  call void @llvm.lifetime.start(i64 32, ptr %ap)
  call void @llvm.va_start(ptr %ap)
  call void @llvm.memcpy.p0.p0.i64(ptr %tmp, ptr %ap, i64 32, i32 8, i1 false)
  call void @baz(ptr %i, ptr nonnull %tmp)
  call void @bar(ptr %buf)
  call void @llvm.va_end(ptr %ap)
  call void @llvm.lifetime.end(i64 32, ptr %ap)
  call void @llvm.lifetime.end(i64 10, ptr %buf)
  ret void
}

declare void @llvm.lifetime.start(i64, ptr nocapture)
declare void @llvm.va_start(ptr)
declare void @baz(ptr, ptr)
declare void @llvm.memcpy.p0.p0.i64(ptr nocapture, ptr nocapture readonly, i64, i32, i1)
declare void @bar(ptr)
declare void @llvm.va_end(ptr)
declare void @llvm.lifetime.end(i64, ptr nocapture)

attributes #0 = { noinline nounwind optnone ssp }

!llvm.module.flags = !{!0}
!0 = !{i32 7, !"direct-access-external-data", i32 1}
