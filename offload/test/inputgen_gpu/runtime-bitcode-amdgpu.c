// Verify AMDGPU runtime bitcode defines the direct-entry state and callback ABI.
// RUN: %clang --target=amdgcn-amd-amdhsa -DINPUTGEN_GPU_RT_DEVICE=1 \
// RUN:   -I%inputgen-gpu-src -I%inputgen-gpu-interface-include \
// RUN:   -I%inputgen-gpu-llvm-include -O3 -std=c11 -nogpulib -nostdlibinc \
// RUN:   -fconvergent-functions -fvisibility=protected -flto -c -emit-llvm \
// RUN:   %inputgen-gpu-src/inputgen_gpu_entry_state.c -o %t.state.bc
// RUN: %clang --target=amdgcn-amd-amdhsa -DINPUTGEN_GPU_RT_DEVICE=1 \
// RUN:   -I%inputgen-gpu-src -I%inputgen-gpu-interface-include \
// RUN:   -I%inputgen-gpu-llvm-include -O3 -std=c11 -nogpulib -nostdlibinc \
// RUN:   -fconvergent-functions -fvisibility=protected -flto -c -emit-llvm \
// RUN:   %inputgen-gpu-src/inputgen_gpu_entry_callbacks.c -o %t.callbacks.bc
// RUN: llvm-link %t.state.bc %t.callbacks.bc -o %t.bc
// RUN: llvm-dis %t.bc -o - | FileCheck %s
// REQUIRES: amdgpu
// REQUIRES: inputgen-gpu-runtime

// CHECK-DAG: @inputgen_buffer = protected {{.*}}global ptr
// CHECK-DAG: @inputgen_buffer_size = protected {{.*}}global i64 0
// CHECK-DAG: @inputgen_buffer_offset = protected {{.*}}global i64 0
// CHECK-DAG: @inputgen_mode = protected {{.*}}global i32 0
// CHECK: define protected {{.*}}i64 @__ig_post_load(i64 {{.*}}, i64 {{.*}}, i32 {{.*}}, i32 {{.*}})
