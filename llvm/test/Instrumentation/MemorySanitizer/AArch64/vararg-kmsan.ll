; RUN: opt < %s -S -passes=msan -msan-kernel=1 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

%struct.__va_list = type { ptr, ptr, ptr, i32, i32 }

define i32 @foo(i32 %guard, ...) {
  %vl = alloca %struct.__va_list, align 8
  call void @llvm.lifetime.start.p0(i64 32, ptr %vl)
  call void @llvm.va_start(ptr %vl)
  call void @llvm.va_end(ptr %vl)
  call void @llvm.lifetime.end.p0(i64 32, ptr %vl)
  ret i32 0
}

; First check if the variadic shadow values are saved in stack with correct
; size (192 is total of general purpose registers size, 64, plus total of
; floating-point registers size, 128).

; CHECK-LABEL: @foo
; CHECK: [[A:%.*]] = load {{.*}} ptr %va_arg_overflow_size
; CHECK: [[B:%.*]] = add i64 192, [[A]]
; CHECK: alloca {{.*}} [[B]]

; We expect three memcpy operations: one for the general purpose registers,
; one for floating-point/SIMD ones, and one for thre remaining arguments.

; Propagate the GR shadow values on for the va_list::__gp_top, adjust the 
; offset in the __msan_va_arg_tls based on va_list:__gp_off, and finally
; issue the memcpy.
; CHECK: [[GRP:%.*]] = getelementptr inbounds i8, ptr {{%.*}}, i64 {{%.*}}
; CHECK: [[GRSIZE:%.*]] = sub i64 64, {{%.*}}
; CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 8 {{%.*}}, ptr align 8 [[GRP]], i64 [[GRSIZE]], i1 false)

; Propagate the VR shadow values on for the va_list::__vr_top, adjust the 
; offset in the __msan_va_arg_tls based on va_list:__vr_off, and finally
; issue the memcpy.
; CHECK: [[VRP:%.*]] = getelementptr inbounds i8, ptr {{%.*}}, i64 {{%.*}}
; CHECK: [[VRSIZE:%.*]] = sub i64 128, {{%.*}}
; CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 8 {{%.*}}, ptr align 8 [[VRP]], i64 [[VRSIZE]], i1 false)

; Copy the remaining shadow values on the va_list::__stack position (it is
; on the constant offset of 192 from __msan_va_arg_tls).
; CHECK: [[STACK:%.*]] = getelementptr inbounds i8, ptr {{%.*}}, i32 192
; CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 16 {{%.*}}, ptr align 16 [[STACK]], i64 {{%.*}}, i1 false)

declare void @llvm.lifetime.start.p0(i64, ptr nocapture) #1
declare void @llvm.va_start(ptr) #2
declare void @llvm.va_end(ptr) #2
declare void @llvm.lifetime.end.p0(i64, ptr nocapture) #1
