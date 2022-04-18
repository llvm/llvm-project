; RUN: opt < %s -S -passes=msan 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

%struct.__va_list = type { i8*, i8*, i8*, i32, i32 }

define i32 @foo(i32 %guard, ...) {
  %vl = alloca %struct.__va_list, align 8
  %1 = bitcast %struct.__va_list* %vl to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* %1)
  call void @llvm.va_start(i8* %1)
  call void @llvm.va_end(i8* %1)
  call void @llvm.lifetime.end.p0i8(i64 32, i8* %1)
  ret i32 0
}

; First check if the variadic shadow values are saved in stack with correct
; size (192 is total of general purpose registers size, 64, plus total of
; floating-point registers size, 128).

; CHECK-LABEL: @foo
; CHECK: [[A:%.*]] = load {{.*}} @__msan_va_arg_overflow_size_tls
; CHECK: [[B:%.*]] = add i64 192, [[A]]
; CHECK: alloca {{.*}} [[B]]

; We expect three memcpy operations: one for the general purpose registers,
; one for floating-point/SIMD ones, and one for thre remaining arguments.

; Propagate the GR shadow values on for the va_list::__gp_top, adjust the 
; offset in the __msan_va_arg_tls based on va_list:__gp_off, and finally
; issue the memcpy.
; CHECK: [[GRP:%.*]] = getelementptr inbounds i8, i8* {{%.*}}, i64 {{%.*}}
; CHECK: [[GRSIZE:%.*]] = sub i64 64, {{%.*}}
; CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 {{%.*}}, i8* align 8 [[GRP]], i64 [[GRSIZE]], i1 false)

; Propagate the VR shadow values on for the va_list::__vr_top, adjust the 
; offset in the __msan_va_arg_tls based on va_list:__vr_off, and finally
; issue the memcpy.
; CHECK: [[VRP:%.*]] = getelementptr inbounds i8, i8* {{%.*}}, i64 {{%.*}}
; CHECK: [[VRSIZE:%.*]] = sub i64 128, {{%.*}}
; CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 {{%.*}}, i8* align 8 [[VRP]], i64 [[VRSIZE]], i1 false)

; Copy the remaining shadow values on the va_list::__stack position (it is
; on the constant offset of 192 from __msan_va_arg_tls).
; CHECK: [[STACK:%.*]] = getelementptr inbounds i8, i8* {{%.*}}, i32 192
; CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 {{%.*}}, i8* align 16 [[STACK]], i64 {{%.*}}, i1 false)

declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1
declare void @llvm.va_start(i8*) #2
declare void @llvm.va_end(i8*) #2
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

define i32 @bar() {
  %1 = call i32 (i32, ...) @foo(i32 0, i32 1, i32 2, double 3.000000e+00, 
                                double 4.000000e+00, i32 5, i32 6,
                                double 7.000000e+00, i32 8, i32 9, i32 10, i32 11)
  ret i32 %1
}

; Save the incoming shadow value from the arguments in the __msan_va_arg_tls
; array.  General purpose registers are saved at positions from 0 to 64, Floating
; point and SIMD are saved from 64 to 192, and the remaining from 192.
; CHECK-LABEL: @bar
; CHECK: store {{.*}} @__msan_va_arg_tls {{.*}} 8
; CHECK: store {{.*}} @__msan_va_arg_tls {{.*}} 16
; CHECK: store {{.*}} @__msan_va_arg_tls {{.*}} 64
; CHECK: store {{.*}} @__msan_va_arg_tls {{.*}} 80
; CHECK: store {{.*}} @__msan_va_arg_tls {{.*}} 24
; CHECK: store {{.*}} @__msan_va_arg_tls {{.*}} 32
; CHECK: store {{.*}} @__msan_va_arg_tls {{.*}} 96
; CHECK: store {{.*}} @__msan_va_arg_tls {{.*}} 40
; CHECK: store {{.*}} @__msan_va_arg_tls {{.*}} 48
; CHECK: store {{.*}} @__msan_va_arg_tls {{.*}} 56
; CHECK: store {{.*}} @__msan_va_arg_tls {{.*}} 192
; CHECK: store {{.*}} 8, {{.*}} @__msan_va_arg_overflow_size_tls

; Test that MSan doesn't generate code overflowing __msan_va_arg_tls when too many arguments are
; passed to a variadic function.

define dso_local i64 @many_args() {
entry:
  %ret = call i64 (i64, ...) @sum(i64 120,
    i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1,
    i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1,
    i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1,
    i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1,
    i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1,
    i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1,
    i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1,
    i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1,
    i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1,
    i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1,
    i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1,
    i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1
  )
  ret i64 %ret
}

; If the size of __msan_va_arg_tls changes the second argument of `add` must also be changed.
; CHECK: i64 add (i64 ptrtoint ([100 x i64]* @__msan_va_arg_tls to i64), i64 792)
; CHECK-NOT: i64 add (i64 ptrtoint ([100 x i64]* @__msan_va_arg_tls to i64), i64 800)
declare i64 @sum(i64 %n, ...)
