// RUN: %clang_cc1 -triple spir -cl-std=CL2.0 %s -finclude-default-header -emit-llvm-bc -o %t.bc
// RUN: llvm-spirv %t.bc -spirv-text -o %t.txt
// RUN: FileCheck < %t.txt %s --check-prefix=CHECK-SPIRV
// RUN: llvm-spirv %t.bc -o %t.spv
// RUN: spirv-val %t.spv
// RUN: llvm-spirv -r %t.spv -o %t.rev.bc
// RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

constant sampler_t constSampl = CLK_FILTER_LINEAR;

__kernel
void sample_kernel(image2d_t input, float2 coords, global float4 *results, sampler_t argSampl) {
  *results = read_imagef(input, constSampl, coords);
  *results = read_imagef(input, argSampl, coords);
  *results = read_imagef(input, CLK_FILTER_NEAREST|CLK_ADDRESS_REPEAT, coords);
}

// CHECK-SPIRV: Capability LiteralSampler
// CHECK-SPIRV: EntryPoint 6 [[sample_kernel:[0-9]+]] "sample_kernel"

// CHECK-SPIRV: TypeSampler [[TypeSampler:[0-9]+]]
// CHECK-SPIRV: TypeSampledImage [[SampledImageTy:[0-9]+]]
// CHECK-SPIRV: ConstantSampler [[TypeSampler]] [[ConstSampler1:[0-9]+]] 0 0 1
// CHECK-SPIRV: ConstantSampler [[TypeSampler]] [[ConstSampler2:[0-9]+]] 3 0 0

// CHECK-SPIRV: Function {{.*}} [[sample_kernel]]
// CHECK-SPIRV: FunctionParameter {{.*}} [[InputImage:[0-9]+]]
// CHECK-SPIRV: FunctionParameter [[TypeSampler]] [[argSampl:[0-9]+]]
// CHECK-LLVM: define spir_kernel void @sample_kernel(%opencl.image2d_ro_t addrspace(1)* %input, <2 x float> %coords, <4 x float> addrspace(1)* nocapture %results, %opencl.sampler_t* %argSampl)

// CHECK-SPIRV: SampledImage [[SampledImageTy]] [[SampledImage1:[0-9]+]] [[InputImage]] [[ConstSampler1]]
// CHECK-SPIRV: ImageSampleExplicitLod {{.*}} [[SampledImage1]]
// CHECK-LLVM:  call spir_func <4 x float> @_Z11read_imagef14ocl_image2d_ro11ocl_samplerDv2_f(%opencl.image2d_ro_t addrspace(1)* %input, %opencl.sampler_t* %0, <2 x float> %coords)

// CHECK-SPIRV: SampledImage [[SampledImageTy]] [[SampledImage2:[0-9]+]] [[InputImage]] [[argSampl]]
// CHECK-SPIRV: ImageSampleExplicitLod {{.*}} [[SampledImage2]]
// CHECK-LLVM:   call spir_func <4 x float> @_Z11read_imagef14ocl_image2d_ro11ocl_samplerDv2_f(%opencl.image2d_ro_t addrspace(1)* %input, %opencl.sampler_t* %argSampl, <2 x float> %coords)

// CHECK-SPIRV: SampledImage [[SampledImageTy]] [[SampledImage3:[0-9]+]] [[InputImage]] [[ConstSampler2]]
// CHECK-SPIRV: ImageSampleExplicitLod {{.*}} [[SampledImage3]]
// CHECK-LLVM: call spir_func <4 x float> @_Z11read_imagef14ocl_image2d_ro11ocl_samplerDv2_f(%opencl.image2d_ro_t addrspace(1)* %input, %opencl.sampler_t* %{{[0-9]+}}, <2 x float> %coords)

// CHECK-LLVM: declare spir_func <4 x float> @_Z11read_imagef14ocl_image2d_ro11ocl_samplerDv2_f(%opencl.image2d_ro_t addrspace(1)*, %opencl.sampler_t*, <2 x float>)
