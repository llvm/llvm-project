; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -print-after-all -o - 2>&1 | FileCheck %s
; RUN: llc -O0 -mtriple=spirv64-amd-amdhsa %s -print-after-all -o - 2>&1 | FileCheck %s

; CHECK: *** IR Dump After SPIRV emit intrinsics (emit-intrinsics) ***

define spir_kernel void @test_pointer_cast(ptr addrspace(1) %src) {
; CHECK-NOT: call{{.*}} void @llvm.spv.assign.ptr.type.p1(ptr addrspace(1) %src, metadata i8 undef, i32 1)
; CHECK: call{{.*}} void @llvm.spv.assign.ptr.type.p1(ptr addrspace(1) %src, metadata i32 poison, i32 1)
  %b = bitcast ptr addrspace(1) %src to ptr addrspace(1)
  %g = getelementptr inbounds i32, ptr addrspace(1) %b, i64 52
  ret void
; CHECK: }
}
