// RUN: %clang_cc1 -O1 -triple spir-unknown-unknown -cl-std=CL2.0 %s -finclude-default-header -emit-llvm-bc -o %t.bc
// RUN: llvm-spirv %t.bc -spirv-text -o %t.txt
// RUN: FileCheck < %t.txt %s --check-prefix=CHECK-SPIRV
// RUN: llvm-spirv %t.bc -o %t.spv
// RUN: spirv-val %t.spv
// RUN: llvm-spirv -r %t.spv -o %t.rev.bc
// RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

void __kernel sample_kernel_read( __global float4 *results,
    read_only image2d_t image,
    sampler_t imageSampler,
    float2 coord,
    float2 dx,
    float2 dy)
{
  *results = read_imagef( image, imageSampler, coord);
  *results = read_imagef( image, imageSampler, coord, 3.14f);
  *results = read_imagef( image, imageSampler, coord, dx, dy);
}

// CHECK-SPIRV: TypeFloat [[float:[0-9]+]] 32
// CHECK-SPIRV: Constant [[float]] [[lodNull:[0-9]+]] 0
// CHECK-SPIRV: Constant [[float]] [[lod:[0-9]+]] 1078523331
// CHECK-SPIRV: FunctionParameter
// CHECK-SPIRV: FunctionParameter
// CHECK-SPIRV: FunctionParameter
// CHECK-SPIRV: FunctionParameter
// CHECK-SPIRV: FunctionParameter {{[0-9]+}} [[dx:[0-9]+]]
// CHECK-SPIRV: FunctionParameter {{[0-9]+}} [[dy:[0-9]+]]

// CHECK-SPIRV: ImageSampleExplicitLod {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} 2 [[lodNull]]
// CHECK-SPIRV: ImageSampleExplicitLod {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} 2 [[lod]]
// CHECK-SPIRV: ImageSampleExplicitLod {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} 4 [[dx]] [[dy]]

// CHECK-LLVM: call spir_func <4 x float> @_Z11read_imagef14ocl_image2d_ro11ocl_samplerDv2_f(%opencl.image2d_ro_t addrspace(1)* %image, %opencl.sampler_t* %imageSampler, <2 x float> %coord)
// CHECK-LLVM: call spir_func <4 x float> @_Z11read_imagef14ocl_image2d_ro11ocl_samplerDv2_ff(%opencl.image2d_ro_t addrspace(1)* %image, %opencl.sampler_t* %imageSampler, <2 x float> %coord, float 0x40091EB860000000)
// CHECK-LLVM: call spir_func <4 x float> @_Z11read_imagef14ocl_image2d_ro11ocl_samplerDv2_fS_S_(%opencl.image2d_ro_t addrspace(1)* %image, %opencl.sampler_t* %imageSampler, <2 x float> %coord, <2 x float> %dx, <2 x float> %dy)
