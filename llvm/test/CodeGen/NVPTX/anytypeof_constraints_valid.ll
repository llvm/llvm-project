; RUN: llvm-as < %s -o /dev/null
; Test cases for int_nvvm_test_anytype intrinsic - VALID combinations

define i32 @test_arg0_i16(i16 %a0, i16 %a1, i32 %a2, ptr %a3, i64 %a4, ptr addrspace(1) %a5) {
  %result = call i32 @llvm.nvvm.test.anytype.i16.i16.i32.p0.i64.p1(i16 %a0, i16 %a1, i32 %a2, ptr %a3, i64 %a4, ptr addrspace(1) %a5)
  ret i32 %result
}

define i32 @test_arg0_i32(i32 %a0, i32 %a1, i32 %a2, ptr %a3, i64 %a4, ptr addrspace(1) %a5) {
  %result = call i32 @llvm.nvvm.test.anytype.i32.i32.i32.p0.i64.p1(i32 %a0, i32 %a1, i32 %a2, ptr %a3, i64 %a4, ptr addrspace(1) %a5)
  ret i32 %result
}

define i32 @test_arg2_anyint(i32 %a0, i32 %a1, i32 %a2, ptr %a3, i64 %a4, ptr addrspace(1) %a5) {
  %result = call i32 @llvm.nvvm.test.anytype.i32.i32.i32.p0.i64.p1(i32 %a0, i32 %a1, i32 %a2, ptr %a3, i64 %a4, ptr addrspace(1) %a5)
  ret i32 %result
}

define i32 @test_arg3_shared_ptr(i32 %a0, i32 %a1, i32 %a2, ptr addrspace(3) %a3, i64 %a4, <2 x i64> %a5) {
  %result = call i32 @llvm.nvvm.test.anytype.i32.i32.i32.p3.i64.v2i64(i32 %a0, i32 %a1, i32 %a2, ptr addrspace(3) %a3, i64 %a4, <2 x i64> %a5)
  ret i32 %result
}

define i32 @test_arg4_double(i32 %a0, i32 %a1, i32 %a2, ptr %a3, double %a4, ptr addrspace(1) %a5) {
  %result = call i32 @llvm.nvvm.test.anytype.i32.i32.i32.p0.f64.p1(i32 %a0, i32 %a1, i32 %a2, ptr %a3, double %a4, ptr addrspace(1) %a5)
  ret i32 %result
}

define i32 @test_arg5_v4i32(i32 %a0, i32 %a1, i32 %a2, ptr %a3, i64 %a4, <2 x i64> %a5) {
  %result = call i32 @llvm.nvvm.test.anytype.i32.i32.i32.p0.i64.v2i64(i32 %a0, i32 %a1, i32 %a2, ptr %a3, i64 %a4, <2 x i64> %a5)
  ret i32 %result
}

define {i32, i32, i64} @test_return_type_valid(i32 %arg) {
  %result = call {i32, i32, i64} @llvm.nvvm.test.return.type.i32(i32 %arg)
  ret {i32, i32, i64} %result
}

declare i32 @llvm.nvvm.test.anytype.i16.i16.i32.p0.i64.p1(i16, i16, i32, ptr, i64, ptr addrspace(1))
declare i32 @llvm.nvvm.test.anytype.i32.i32.i32.p0.i64.p1(i32, i32, i32, ptr, i64, ptr addrspace(1))
declare i32 @llvm.nvvm.test.anytype.i32.i32.f32.p0.i64.p1(i32, i32, i32, ptr, i64, ptr addrspace(1))
declare i32 @llvm.nvvm.test.anytype.i32.i32.i32.p3.i64.v2i64(i32, i32, i32, ptr addrspace(3), i64, <2 x i64>)
declare i32 @llvm.nvvm.test.anytype.i32.i32.i32.p0.f64.p1(i32, i32, i32, ptr, double, ptr addrspace(1))
declare i32 @llvm.nvvm.test.anytype.i32.i32.i32.p0.i64.v2i64(i32, i32, i32, ptr, i64, <2 x i64>)
declare {i32, i32, i64} @llvm.nvvm.test.return.type.i32(i32)
