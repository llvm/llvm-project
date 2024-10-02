; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

declare ptr @llvm.returnaddress(i32)
define void @return_address(i32 %var) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %var
  ; CHECK-NEXT: %result = call ptr @llvm.returnaddress(i32 %var)
  %result = call ptr @llvm.returnaddress(i32 %var)
  ret void
}

declare ptr @llvm.frameaddress(i32)
define void @frame_address(i32 %var) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %var
  ; CHECK-NEXT: %result = call ptr @llvm.frameaddress.p0(i32 %var)
  %result = call ptr @llvm.frameaddress(i32 %var)
  ret void
}

declare void @llvm.memcpy.p0.p0.i32(ptr nocapture, ptr nocapture, i32, i1)
define void @memcpy(ptr %dest, ptr %src, i1 %is.volatile) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i1 %is.volatile
  ; CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr %dest, ptr %src, i32 8, i1 %is.volatile)
  call void @llvm.memcpy.p0.p0.i32(ptr %dest, ptr %src, i32 8, i1 %is.volatile)
  ret void
}

declare void @llvm.memcpy.inline.p0.p0.i32(ptr nocapture, ptr nocapture, i32, i1)
define void @memcpy_inline_is_volatile(ptr %dest, ptr %src, i1 %is.volatile) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i1 %is.volatile
  ; CHECK-NEXT: call void @llvm.memcpy.inline.p0.p0.i32(ptr %dest, ptr %src, i32 8, i1 %is.volatile)
  call void @llvm.memcpy.inline.p0.p0.i32(ptr %dest, ptr %src, i32 8, i1 %is.volatile)
  ret void
}

declare void @llvm.memmove.p0.p0.i32(ptr nocapture, ptr nocapture, i32, i1)
define void @memmove(ptr %dest, ptr %src, i1 %is.volatile) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i1 %is.volatile
  ; CHECK-NEXT: call void @llvm.memmove.p0.p0.i32(ptr %dest, ptr %src, i32 8, i1 %is.volatile)
  call void @llvm.memmove.p0.p0.i32(ptr %dest, ptr %src, i32 8, i1 %is.volatile)
  ret void
}

declare void @llvm.memset.p0.i32(ptr nocapture, i8, i32, i1)
define void @memset(ptr %dest, i8 %val, i1 %is.volatile) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i1 %is.volatile
  ; CHECK-NEXT: call void @llvm.memset.p0.i32(ptr %dest, i8 %val, i32 8, i1 %is.volatile)
  call void @llvm.memset.p0.i32(ptr %dest, i8 %val, i32 8, i1 %is.volatile)
  ret void
}

declare void @llvm.memset.inline.p0.i32(ptr nocapture, i8, i32, i1)
define void @memset_inline_is_volatile(ptr %dest, i8 %value, i1 %is.volatile) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i1 %is.volatile
  ; CHECK-NEXT: call void @llvm.memset.inline.p0.i32(ptr %dest, i8 %value, i32 8, i1 %is.volatile)
  call void @llvm.memset.inline.p0.i32(ptr %dest, i8 %value, i32 8, i1 %is.volatile)
  ret void
}


declare i64 @llvm.objectsize.i64.p0(ptr, i1, i1, i1)
define void @objectsize(ptr %ptr, i1 %a, i1 %b, i1 %c) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i1 %a
  ; CHECK-NEXT: %val0 = call i64 @llvm.objectsize.i64.p0(ptr %ptr, i1 %a, i1 false, i1 false)
  %val0 = call i64 @llvm.objectsize.i64.p0(ptr %ptr, i1 %a, i1 false, i1 false)

  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i1 %b
  ; CHECK-NEXT: %val1 = call i64 @llvm.objectsize.i64.p0(ptr %ptr, i1 false, i1 %b, i1 false)
  %val1 = call i64 @llvm.objectsize.i64.p0(ptr %ptr, i1 false, i1 %b, i1 false)

  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i1 %c
  ; CHECK-NEXT: %val2 = call i64 @llvm.objectsize.i64.p0(ptr %ptr, i1 false, i1 false, i1 %c)
  %val2 = call i64 @llvm.objectsize.i64.p0(ptr %ptr, i1 false, i1 false, i1 %c)
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

declare i64 @llvm.smul.fix.sat.i64(i64, i64, i32)
define i64 @smul_fix_sat(i64 %arg0, i64 %arg1, i32 %arg2) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg2
  ; CHECK-NEXT: %ret = call i64 @llvm.smul.fix.sat.i64(i64 %arg0, i64 %arg1, i32 %arg2)
  %ret = call i64 @llvm.smul.fix.sat.i64(i64 %arg0, i64 %arg1, i32 %arg2)
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

declare i64 @llvm.umul.fix.sat.i64(i64, i64, i32)
define i64 @umul_fix_sat(i64 %arg0, i64 %arg1, i32 %arg2) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg2
  ; CHECK-NEXT: %ret = call i64 @llvm.umul.fix.sat.i64(i64 %arg0, i64 %arg1, i32 %arg2)
  %ret = call i64 @llvm.umul.fix.sat.i64(i64 %arg0, i64 %arg1, i32 %arg2)
  ret i64 %ret
}

declare <2 x double> @llvm.masked.load.v2f64.p0(ptr, i32, <2 x i1>, <2 x double>)
define <2 x double> @masked_load(<2 x i1> %mask, ptr %addr, <2 x double> %dst, i32 %align) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %align
  ; CHECK-NEXT: %res = call <2 x double> @llvm.masked.load.v2f64.p0(ptr %addr, i32 %align, <2 x i1> %mask, <2 x double> %dst)
  %res = call <2 x double> @llvm.masked.load.v2f64.p0(ptr %addr, i32 %align, <2 x i1> %mask, <2 x double> %dst)
  ret <2 x double> %res
}

declare void @llvm.masked.store.v4i32.p0(<4 x i32>, ptr, i32, <4 x i1>)
define void @masked_store(<4 x i1> %mask, ptr %addr, <4 x i32> %val, i32 %align) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %align
  ; CHECK-NEXT: call void @llvm.masked.store.v4i32.p0(<4 x i32> %val, ptr %addr, i32 %align, <4 x i1> %mask)
  call void @llvm.masked.store.v4i32.p0(<4 x i32> %val, ptr %addr, i32 %align, <4 x i1> %mask)
  ret void
}

declare <2 x double> @llvm.masked.gather.v2f64.v2p0(<2 x ptr>, i32, <2 x i1>, <2 x double>)
define <2 x double> @test_gather(<2 x ptr> %ptrs, <2 x i1> %mask, <2 x double> %src0, i32 %align)  {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK: i32 %align
  ; CHECK: %res = call <2 x double> @llvm.masked.gather.v2f64.v2p0(<2 x ptr> %ptrs, i32 %align, <2 x i1> %mask, <2 x double> %src0)
  %res = call <2 x double> @llvm.masked.gather.v2f64.v2p0(<2 x ptr> %ptrs, i32 %align, <2 x i1> %mask, <2 x double> %src0)
  ret <2 x double> %res
}

declare void @llvm.masked.scatter.v8i32.v8p0(<8 x i32>, <8 x ptr>, i32, <8 x i1>)
define void @test_scatter_8i32(<8 x i32> %a1, <8 x ptr> %ptr, <8 x i1> %mask, i32 %align) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %align
  ; CHECK-NEXT: call void @llvm.masked.scatter.v8i32.v8p0(<8 x i32> %a1, <8 x ptr> %ptr, i32 %align, <8 x i1> %mask)
  call void @llvm.masked.scatter.v8i32.v8p0(<8 x i32> %a1, <8 x ptr> %ptr, i32 %align, <8 x i1> %mask)
  ret void
}

declare void @llvm.lifetime.start.p0(i64, ptr)
define void @test_lifetime_start(i64 %arg0, ptr %ptr) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i64 %arg0
  ; CHECK-NEXT: call void @llvm.lifetime.start.p0(i64 %arg0, ptr %ptr)
  call void @llvm.lifetime.start.p0(i64 %arg0, ptr %ptr)
  ret void
}

declare void @llvm.lifetime.end.p0(i64, ptr)
define void @test_lifetime_end(i64 %arg0, ptr %ptr) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i64 %arg0
  ; CHECK-NEXT: call void @llvm.lifetime.end.p0(i64 %arg0, ptr %ptr)
  call void @llvm.lifetime.end.p0(i64 %arg0, ptr %ptr)
  ret void
}

declare void @llvm.invariant.start.p0(i64, ptr)
define void @test_invariant_start(i64 %arg0, ptr %ptr) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i64 %arg0
  ; CHECK-NEXT: call void @llvm.invariant.start.p0(i64 %arg0, ptr %ptr)
  call void @llvm.invariant.start.p0(i64 %arg0, ptr %ptr)
  ret void
}

declare void @llvm.invariant.end.p0(ptr, i64, ptr)
define void @test_invariant_end(ptr %scope, i64 %arg1, ptr %ptr) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i64 %arg1
  ; CHECK-NEXT: call void @llvm.invariant.end.p0(ptr %scope, i64 %arg1, ptr %ptr)
  call void @llvm.invariant.end.p0(ptr %scope, i64 %arg1, ptr %ptr)
  ret void
}

declare void @llvm.prefetch(ptr, i32, i32, i32)
define void @test_prefetch(ptr %ptr, i32 %arg0, i32 %arg1) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg0
  ; CHECK-NEXT: call void @llvm.prefetch.p0(ptr %ptr, i32 %arg0, i32 0, i32 0)
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT:  i32 %arg1
  call void @llvm.prefetch(ptr %ptr, i32 %arg0, i32 0, i32 0)
  call void @llvm.prefetch(ptr %ptr, i32 0, i32 %arg1, i32 0)
  ret void
}

declare void @llvm.localrecover(ptr, ptr, i32)
define void @test_localrecover(ptr %func, ptr %fp, i32 %idx) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %idx
  ; CHECK-NEXT: call void @llvm.localrecover(ptr %func, ptr %fp, i32 %idx)
  call void @llvm.localrecover(ptr %func, ptr %fp, i32 %idx)
  ret void
}

declare token @llvm.experimental.gc.statepoint.p0(i64, i32, ptr, i32, i32, ...)

define private void @f() {
  ret void
}

define void @calls_statepoint(ptr addrspace(1) %arg0, i64 %arg1, i32 %arg2, i32 %arg4, i32 %arg5) gc "statepoint-example" {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i64 %arg1
  ; CHECK-NEXT: %safepoint0 = call token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(i64 %arg1, i32 0, ptr @f, i32 0, i32 0, i32 0, i32 5, i32 0, i32 0, i32 0, i32 10, i32 0, ptr addrspace(1) %arg0, ptr addrspace(1) %arg0, ptr addrspace(1) %arg0, ptr addrspace(1) %arg0)
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg2
  ; CHECK-NEXT: %safepoint1 = call token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(i64 0, i32 %arg2, ptr @f, i32 0, i32 0, i32 0, i32 5, i32 0, i32 0, i32 0, i32 10, i32 0, ptr addrspace(1) %arg0, ptr addrspace(1) %arg0, ptr addrspace(1) %arg0, ptr addrspace(1) %arg0)
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg4
  ; CHECK-NEXT: %safepoint2 = call token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(i64 0, i32 0, ptr @f, i32 %arg4, i32 0, i32 0, i32 5, i32 0, i32 0, i32 0, i32 10, i32 0, ptr addrspace(1) %arg0, ptr addrspace(1) %arg0, ptr addrspace(1) %arg0, ptr addrspace(1) %arg0)
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg5
  ; CHECK-NEXT: %safepoint3 = call token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(i64 0, i32 0, ptr @f, i32 0, i32 %arg5, i32 0, i32 5, i32 0, i32 0, i32 0, i32 10, i32 0, ptr addrspace(1) %arg0, ptr addrspace(1) %arg0, ptr addrspace(1) %arg0, ptr addrspace(1) %arg0)
  %safepoint0 = call token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(i64 %arg1, i32 0, ptr @f, i32 0, i32 0, i32 0, i32 5, i32 0, i32 0, i32 0, i32 10, i32 0, ptr addrspace(1) %arg0, ptr addrspace(1) %arg0, ptr addrspace(1) %arg0, ptr addrspace(1) %arg0)
  %safepoint1 = call token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(i64 0, i32 %arg2, ptr @f, i32 0, i32 0, i32 0, i32 5, i32 0, i32 0, i32 0, i32 10, i32 0, ptr addrspace(1) %arg0, ptr addrspace(1) %arg0, ptr addrspace(1) %arg0, ptr addrspace(1) %arg0)
  %safepoint2 = call token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(i64 0, i32 0, ptr @f, i32 %arg4, i32 0, i32 0, i32 5, i32 0, i32 0, i32 0, i32 10, i32 0, ptr addrspace(1) %arg0, ptr addrspace(1) %arg0, ptr addrspace(1) %arg0, ptr addrspace(1) %arg0)
  %safepoint3 = call token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(i64 0, i32 0, ptr @f, i32 0, i32 %arg5, i32 0, i32 5, i32 0, i32 0, i32 0, i32 10, i32 0, ptr addrspace(1) %arg0, ptr addrspace(1) %arg0, ptr addrspace(1) %arg0, ptr addrspace(1) %arg0)
  ret void
}

declare void @llvm.experimental.patchpoint.void(i64, i32, ptr, i32, ...)
declare i64 @llvm.experimental.patchpoint.i64(i64, i32, ptr, i32, ...)

define void @test_patchpoint(i64 %arg0, i32 %arg1, i32 %arg2) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i64 %arg0
  ; CHECK-NEXT: call void (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.void(i64 %arg0, i32 4, ptr null, i32 0)
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg1
  ; CHECK-NEXT: call void (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.void(i64 0, i32 %arg1, ptr null, i32 0)
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg2
  ; CHECK-NEXT: call void (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.void(i64 0, i32 4, ptr null, i32 %arg2)
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i64 %arg0
  ; CHECK-NEXT: %patchpoint0 = call i64 (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.i64(i64 %arg0, i32 4, ptr null, i32 0)
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg1
  ; CHECK-NEXT: %patchpoint1 = call i64 (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.i64(i64 0, i32 %arg1, ptr null, i32 0)
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg2
  ; CHECK-NEXT: %patchpoint2 = call i64 (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.i64(i64 0, i32 4, ptr null, i32 %arg2)
  call void (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.void(i64 %arg0, i32 4, ptr null, i32 0)
  call void (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.void(i64 0, i32 %arg1, ptr null, i32 0)
  call void (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.void(i64 0, i32 4, ptr null, i32 %arg2)
  %patchpoint0 = call i64 (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.i64(i64 %arg0, i32 4, ptr null, i32 0)
  %patchpoint1 = call i64 (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.i64(i64 0, i32 %arg1, ptr null, i32 0)
  %patchpoint2 = call i64 (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.i64(i64 0, i32 4, ptr null, i32 %arg2)
  ret void
}

declare void @llvm.hwasan.check.memaccess(ptr, ptr, i32)

define void @hwasan_check_memaccess(ptr %arg0,ptr %arg1, i32 %arg2) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK: i32 %arg2
  ; CHECK: call void @llvm.hwasan.check.memaccess(ptr %arg0, ptr %arg1, i32 %arg2)
  call void @llvm.hwasan.check.memaccess(ptr %arg0,ptr %arg1, i32 %arg2)
  ret void
}

declare void @llvm.eh.sjlj.callsite(i32)

define void @eh_sjlj_callsite(i32 %arg0) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK: i32 %arg0
  ; CHECK: call void @llvm.eh.sjlj.callsite(i32 %arg0)
  call void @llvm.eh.sjlj.callsite(i32 %arg0)
  ret void
}
