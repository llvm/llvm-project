; RUN: not llc -O0 -mtriple=spirv32-unknown-unknown %s -o %t.spvt 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR
; CHECK-ERROR: LLVM ERROR: Implicit binding calls with the same order ID must have the same descriptor set

@.str = private unnamed_addr constant [2 x i8] c"b\00", align 1
@.str.2 = private unnamed_addr constant [2 x i8] c"c\00", align 1

define void @main() local_unnamed_addr #0 {
entry:
  %0 = tail call target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 0) @llvm.spv.resource.handlefromimplicitbinding.tspirv.SignedImage_i32_5_2_0_0_2_0t(i32 0, i32 0, i32 1, i32 0, ptr nonnull @.str)
  %1 = tail call noundef align 4 dereferenceable(4) ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.SignedImage_i32_5_2_0_0_2_0t(target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 0) %0, i32 0)
  %2 = load i32, ptr addrspace(11) %1, align 4
  %3 = tail call target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 0) @llvm.spv.resource.handlefromimplicitbinding.tspirv.SignedImage_i32_5_2_0_0_2_0t(i32 0, i32 1, i32 1, i32 0, ptr nonnull @.str.2)
  %4 = tail call noundef align 4 dereferenceable(4) ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.SignedImage_i32_5_2_0_0_2_0t(target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 0) %3, i32 0)
  store i32 %2, ptr addrspace(11) %4, align 4
  ret void
}


attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
