// RUN: %clang_cc1 -O1 -triple spir-unknown-unknown -cl-std=CL2.0 %s -finclude-default-header -emit-llvm-bc -o %t.bc
// RUN: llvm-spirv %t.bc -spirv-text -o %t.txt
// RUN: FileCheck < %t.txt %s --check-prefix=CHECK-SPIRV
// RUN: llvm-spirv %t.bc -o %t.spv
// RUN: llvm-spirv -r %t.spv -o %t.rev.bc
// RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

void sample_kernel_write(float4 input, write_only image2d_t output, int2 coord)
{
  write_imagef( output, coord , 5, input);
  write_imagef( output, coord , input);
}

// CHECK-SPIRV: Constant {{[0-9]+}} [[lod:[0-9]+]] 5
// CHECK-SPIRV: ImageWrite [[image:[0-9]+]] [[coord:[0-9]+]] [[texel:[0-9]+]] 2 [[lod]]
// CHECK-SPIRV: ImageWrite [[image]] [[coord]] [[texel]]

// CHECK-LLVM: call spir_func void @_Z12write_imagef14ocl_image2d_woDv2_iiDv4_f(%opencl.image2d_wo_t addrspace(1)* %output, <2 x i32> %coord, i32 5, <4 x float> %input)
// CHECK-LLVM: call spir_func void @_Z12write_imagef14ocl_image2d_woDv2_iDv4_f(%opencl.image2d_wo_t addrspace(1)* %output, <2 x i32> %coord, <4 x float> %input)
