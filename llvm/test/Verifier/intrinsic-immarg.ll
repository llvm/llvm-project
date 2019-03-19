; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

declare i8* @llvm.returnaddress(i32)
define void @return_address(i32 %var) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %var
  ; CHECK-NEXT: %result = call i8* @llvm.returnaddress(i32 %var)
  %result = call i8* @llvm.returnaddress(i32 %var)
  ret void
}

declare i8* @llvm.frameaddress(i32)
define void @frame_address(i32 %var) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %var
  ; CHECK-NEXT: %result = call i8* @llvm.frameaddress(i32 %var)
  %result = call i8* @llvm.frameaddress(i32 %var)
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i1)
define void @memcpy(i8* %dest, i8* %src, i1 %is.volatile) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i1 %is.volatile
  ; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dest, i8* %src, i32 8, i1 %is.volatile)
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dest, i8* %src, i32 8, i1 %is.volatile)
  ret void
}

declare void @llvm.memmove.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i1)
define void @memmove(i8* %dest, i8* %src, i1 %is.volatile) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i1 %is.volatile
  ; CHECK-NEXT: call void @llvm.memmove.p0i8.p0i8.i32(i8* %dest, i8* %src, i32 8, i1 %is.volatile)
  call void @llvm.memmove.p0i8.p0i8.i32(i8* %dest, i8* %src, i32 8, i1 %is.volatile)
  ret void
}

declare void @llvm.memset.p0i8.i32(i8* nocapture, i8, i32, i1)
define void @memset(i8* %dest, i8 %val, i1 %is.volatile) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i1 %is.volatile
  ; CHECK-NEXT: call void @llvm.memset.p0i8.i32(i8* %dest, i8 %val, i32 8, i1 %is.volatile)
  call void @llvm.memset.p0i8.i32(i8* %dest, i8 %val, i32 8, i1 %is.volatile)
  ret void
}


declare i64 @llvm.objectsize.i64.p0i8(i8*, i1, i1, i1)
define void @objectsize(i8* %ptr, i1 %a, i1 %b, i1 %c) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i1 %a
  ; CHECK-NEXT: %val0 = call i64 @llvm.objectsize.i64.p0i8(i8* %ptr, i1 %a, i1 false, i1 false)
  %val0 = call i64 @llvm.objectsize.i64.p0i8(i8* %ptr, i1 %a, i1 false, i1 false)

  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i1 %b
  ; CHECK-NEXT: %val1 = call i64 @llvm.objectsize.i64.p0i8(i8* %ptr, i1 false, i1 %b, i1 false)
  %val1 = call i64 @llvm.objectsize.i64.p0i8(i8* %ptr, i1 false, i1 %b, i1 false)

  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i1 %c
  ; CHECK-NEXT: %val2 = call i64 @llvm.objectsize.i64.p0i8(i8* %ptr, i1 false, i1 false, i1 %c)
  %val2 = call i64 @llvm.objectsize.i64.p0i8(i8* %ptr, i1 false, i1 false, i1 %c)
  ret void
}

declare i64 @llvm.smul.fix.i64(i64, i64, i32)
define i64 @smul_fix(i64 %arg0, i64 %arg1, i32 %arg2) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg2
  ; CHECK-NEXT: %ret = call i64 @llvm.smul.fix.i64(i64 %arg0, i64 %arg1, i32 %arg2)
  %ret = call i64 @llvm.smul.fix.i64(i64 %arg0, i64 %arg1, i32 %arg2)
  ret i64 %ret
}

declare i64 @llvm.umul.fix.i64(i64, i64, i32)
define i64 @umul_fix(i64 %arg0, i64 %arg1, i32 %arg2) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg2
  ; CHECK-NEXT: %ret = call i64 @llvm.umul.fix.i64(i64 %arg0, i64 %arg1, i32 %arg2)
  %ret = call i64 @llvm.umul.fix.i64(i64 %arg0, i64 %arg1, i32 %arg2)
  ret i64 %ret
}

declare <2 x double> @llvm.masked.load.v2f64.p0v2f64(<2 x double>*, i32, <2 x i1>, <2 x double>)
define <2 x double> @masked_load(<2 x i1> %mask, <2 x double>* %addr, <2 x double> %dst, i32 %align) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %align
  ; CHECK-NEXT: %res = call <2 x double> @llvm.masked.load.v2f64.p0v2f64(<2 x double>* %addr, i32 %align, <2 x i1> %mask, <2 x double> %dst)
  %res = call <2 x double> @llvm.masked.load.v2f64.p0v2f64(<2 x double>* %addr, i32 %align, <2 x i1> %mask, <2 x double> %dst)
  ret <2 x double> %res
}

declare void @llvm.masked.store.v4i32.p0v4i32(<4 x i32>, <4 x i32>*, i32, <4 x i1>)
define void @masked_store(<4 x i1> %mask, <4 x i32>* %addr, <4 x i32> %val, i32 %align) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %align
  ; CHECK-NEXT: call void @llvm.masked.store.v4i32.p0v4i32(<4 x i32> %val, <4 x i32>* %addr, i32 %align, <4 x i1> %mask)
  call void @llvm.masked.store.v4i32.p0v4i32(<4 x i32> %val, <4 x i32>* %addr, i32 %align, <4 x i1> %mask)
  ret void
}

declare <2 x double> @llvm.masked.gather.v2f64.v2p0f64(<2 x double*>, i32, <2 x i1>, <2 x double>)
define <2 x double> @test_gather(<2 x double*> %ptrs, <2 x i1> %mask, <2 x double> %src0, i32 %align)  {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK: i32 %align
  ; CHECK: %res = call <2 x double> @llvm.masked.gather.v2f64.v2p0f64(<2 x double*> %ptrs, i32 %align, <2 x i1> %mask, <2 x double> %src0)
  %res = call <2 x double> @llvm.masked.gather.v2f64.v2p0f64(<2 x double*> %ptrs, i32 %align, <2 x i1> %mask, <2 x double> %src0)
  ret <2 x double> %res
}

declare void @llvm.masked.scatter.v8i32.v8p0i32(<8 x i32>, <8 x i32*>, i32, <8 x i1>)
define void @test_scatter_8i32(<8 x i32> %a1, <8 x i32*> %ptr, <8 x i1> %mask, i32 %align) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %align
  ; CHECK-NEXT: call void @llvm.masked.scatter.v8i32.v8p0i32(<8 x i32> %a1, <8 x i32*> %ptr, i32 %align, <8 x i1> %mask)
  call void @llvm.masked.scatter.v8i32.v8p0i32(<8 x i32> %a1, <8 x i32*> %ptr, i32 %align, <8 x i1> %mask)
  ret void
}

declare void @llvm.lifetime.start.p0i8(i64, i8*)
define void @test_lifetime_start(i64 %arg0, i8* %ptr) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i64 %arg0
  ; CHECK-NEXT: call void @llvm.lifetime.start.p0i8(i64 %arg0, i8* %ptr)
  call void @llvm.lifetime.start.p0i8(i64 %arg0, i8* %ptr)
  ret void
}

declare void @llvm.lifetime.end.p0i8(i64, i8*)
define void @test_lifetime_end(i64 %arg0, i8* %ptr) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i64 %arg0
  ; CHECK-NEXT: call void @llvm.lifetime.end.p0i8(i64 %arg0, i8* %ptr)
  call void @llvm.lifetime.end.p0i8(i64 %arg0, i8* %ptr)
  ret void
}

declare void @llvm.invariant.start.p0i8(i64, i8*)
define void @test_invariant_start(i64 %arg0, i8* %ptr) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i64 %arg0
  ; CHECK-NEXT: call void @llvm.invariant.start.p0i8(i64 %arg0, i8* %ptr)
  call void @llvm.invariant.start.p0i8(i64 %arg0, i8* %ptr)
  ret void
}

declare void @llvm.invariant.end.p0i8({}*, i64, i8*)
define void @test_invariant_end({}* %scope, i64 %arg1, i8* %ptr) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i64 %arg1
  ; CHECK-NEXT: call void @llvm.invariant.end.p0i8({}* %scope, i64 %arg1, i8* %ptr)
  call void @llvm.invariant.end.p0i8({}* %scope, i64 %arg1, i8* %ptr)
  ret void
}

declare void @llvm.prefetch(i8*, i32, i32, i32)
define void @test_prefetch(i8* %ptr, i32 %arg0, i32 %arg1) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg0
  ; CHECK-NEXT: call void @llvm.prefetch(i8* %ptr, i32 %arg0, i32 0, i32 0)
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT:  i32 %arg1
  call void @llvm.prefetch(i8* %ptr, i32 %arg0, i32 0, i32 0)
  call void @llvm.prefetch(i8* %ptr, i32 0, i32 %arg1, i32 0)
  ret void
}

declare void @llvm.localrecover(i8*, i8*, i32)
define void @test_localrecover(i8* %func, i8* %fp, i32 %idx) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %idx
  ; CHECK-NEXT: call void @llvm.localrecover(i8* %func, i8* %fp, i32 %idx)
  call void @llvm.localrecover(i8* %func, i8* %fp, i32 %idx)
  ret void
}

declare token @llvm.experimental.gc.statepoint.p0f_isVoidf(i64, i32, void ()*, i32, i32, ...)

define private void @f() {
  ret void
}

define void @calls_statepoint(i8 addrspace(1)* %arg0, i64 %arg1, i32 %arg2, i32 %arg4, i32 %arg5) gc "statepoint-example" {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i64 %arg1
  ; CHECK-NEXT: %safepoint0 = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 %arg1, i32 0, void ()* @f, i32 0, i32 0, i32 0, i32 5, i32 0, i32 0, i32 0, i32 10, i32 0, i8 addrspace(1)* %arg0, i64 addrspace(1)* %cast, i8 addrspace(1)* %arg0, i8 addrspace(1)* %arg0)
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg2
  ; CHECK-NEXT: %safepoint1 = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 %arg2, void ()* @f, i32 0, i32 0, i32 0, i32 5, i32 0, i32 0, i32 0, i32 10, i32 0, i8 addrspace(1)* %arg0, i64 addrspace(1)* %cast, i8 addrspace(1)* %arg0, i8 addrspace(1)* %arg0)
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg4
  ; CHECK-NEXT: %safepoint2 = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @f, i32 %arg4, i32 0, i32 0, i32 5, i32 0, i32 0, i32 0, i32 10, i32 0, i8 addrspace(1)* %arg0, i64 addrspace(1)* %cast, i8 addrspace(1)* %arg0, i8 addrspace(1)* %arg0)
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg5
  ; CHECK-NEXT: %safepoint3 = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @f, i32 0, i32 %arg5, i32 0, i32 5, i32 0, i32 0, i32 0, i32 10, i32 0, i8 addrspace(1)* %arg0, i64 addrspace(1)* %cast, i8 addrspace(1)* %arg0, i8 addrspace(1)* %arg0)
  %cast = bitcast i8 addrspace(1)* %arg0 to i64 addrspace(1)*
  %safepoint0 = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 %arg1, i32 0, void ()* @f, i32 0, i32 0, i32 0, i32 5, i32 0, i32 0, i32 0, i32 10, i32 0, i8 addrspace(1)* %arg0, i64 addrspace(1)* %cast, i8 addrspace(1)* %arg0, i8 addrspace(1)* %arg0)
  %safepoint1 = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 %arg2, void ()* @f, i32 0, i32 0, i32 0, i32 5, i32 0, i32 0, i32 0, i32 10, i32 0, i8 addrspace(1)* %arg0, i64 addrspace(1)* %cast, i8 addrspace(1)* %arg0, i8 addrspace(1)* %arg0)
  %safepoint2 = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @f, i32 %arg4, i32 0, i32 0, i32 5, i32 0, i32 0, i32 0, i32 10, i32 0, i8 addrspace(1)* %arg0, i64 addrspace(1)* %cast, i8 addrspace(1)* %arg0, i8 addrspace(1)* %arg0)
  %safepoint3 = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @f, i32 0, i32 %arg5, i32 0, i32 5, i32 0, i32 0, i32 0, i32 10, i32 0, i8 addrspace(1)* %arg0, i64 addrspace(1)* %cast, i8 addrspace(1)* %arg0, i8 addrspace(1)* %arg0)
  ret void
}

declare void @llvm.hwasan.check.memaccess(i8*, i8*, i32)

define void @hwasan_check_memaccess(i8* %arg0,i8* %arg1, i32 %arg2) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK: i32 %arg2
  ; CHECK: call void @llvm.hwasan.check.memaccess(i8* %arg0, i8* %arg1, i32 %arg2)
  call void @llvm.hwasan.check.memaccess(i8* %arg0,i8* %arg1, i32 %arg2)
  ret void
}
