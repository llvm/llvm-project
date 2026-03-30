; RUN: not llvm-as < %s -o /dev/null
; Test cases for int_nvvm_test_anytype intrinsic - INVALID combinations

; arg0 must be i16 or i32, not i8
define i32 @invalid_arg0_i8(i8 %a0, i8 %a1, i32 %a2, ptr %a3, i64 %a4, ptr addrspace(1) %a5) {
  %result = call i32 @llvm.nvvm.test.anytype.i8.i8.i32.p0.i64.p1(i8 %a0, i8 %a1, i32 %a2, ptr %a3, i64 %a4, ptr addrspace(1) %a5)
  ret i32 %result
}

; arg1 must match arg0 (i16), not i32
define i32 @invalid_arg1_mismatch_i16_i32(i16 %a0, i32 %a1, i32 %a2, ptr %a3, i64 %a4, ptr addrspace(1) %a5) {
  %result = call i32 @llvm.nvvm.test.anytype.i16.i32.i32.p0.i64.p1(i16 %a0, i32 %a1, i32 %a2, ptr %a3, i64 %a4, ptr addrspace(1) %a5)
  ret i32 %result
}

; arg2 must be anyint or float, not double
define i32 @invalid_arg2_double(i32 %a0, i32 %a1, double %a2, ptr %a3, i64 %a4, ptr addrspace(1) %a5) {
  %result = call i32 @llvm.nvvm.test.anytype.i32.i32.f64.p0.i64.p1(i32 %a0, i32 %a1, double %a2, ptr %a3, i64 %a4, ptr addrspace(1) %a5)
  ret i32 %result
}

; arg3 must be ptr or ptr addrspace(3), not ptr addrspace(1) (global)
define i32 @invalid_arg3_ptr_as1_wrong(i32 %a0, i32 %a1, i32 %a2, ptr addrspace(1) %a3, i64 %a4, <2 x i64> %a5) {
  %result = call i32 @llvm.nvvm.test.anytype.i32.i32.i32.p1.i64.v2i64(i32 %a0, i32 %a1, i32 %a2, ptr addrspace(1) %a3, i64 %a4, <2 x i64> %a5)
  ret i32 %result
}

; arg4 must be i64 or double, not i32
define i32 @invalid_arg4_i32(i32 %a0, i32 %a1, i32 %a2, ptr %a3, i32 %a4, ptr addrspace(1) %a5) {
  %result = call i32 @llvm.nvvm.test.anytype.i32.i32.i32.p0.i32.p1(i32 %a0, i32 %a1, i32 %a2, ptr %a3, i32 %a4, ptr addrspace(1) %a5)
  ret i32 %result
}

; arg5 must be ptr addrspace(1) or <2 x i64>, not <2 x i32>
define i32 @invalid_arg5_v2i32(i32 %a0, i32 %a1, i32 %a2, ptr %a3, i64 %a4, <2 x i32> %a5) {
  %result = call i32 @llvm.nvvm.test.anytype.i32.i32.i32.p0.i64.v2i32(i32 %a0, i32 %a1, i32 %a2, ptr %a3, i64 %a4, <2 x i32> %a5)
  ret i32 %result
}

define {i32, i8, i64} @invalid_return_type_i8(i32 %arg) {
  %result = call {i32, i8, i64} @llvm.nvvm.test.return.type.i8(i32 %arg)
  ret {i32, i8, i64} %result
}

declare i32 @llvm.nvvm.test.anytype.i8.i8.i32.p0.i64.p1(i8, i8, i32, ptr, i64, ptr addrspace(1))
declare i32 @llvm.nvvm.test.anytype.i16.i32.i32.p0.i64.p1(i16, i32, i32, ptr, i64, ptr addrspace(1))
declare i32 @llvm.nvvm.test.anytype.i32.i32.f64.p0.i64.p1(i32, i32, double, ptr, i64, ptr addrspace(1))
declare i32 @llvm.nvvm.test.anytype.i32.i32.i32.p1.i64.v2i64(i32, i32, i32, ptr addrspace(1), i64, <2 x i64>)
declare i32 @llvm.nvvm.test.anytype.i32.i32.i32.p0.i32.p1(i32, i32, i32, ptr, i32, ptr addrspace(1))
declare i32 @llvm.nvvm.test.anytype.i32.i32.i32.p0.i64.v2i32(i32, i32, i32, ptr, i64, <2 x i32>)
declare {i32, i8, i64} @llvm.nvvm.test.return.type.i8(i32)
