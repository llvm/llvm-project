// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s
 
// Checking the translation of the `gpu.binary` & `gpu.launch_fun` ops.
module attributes {gpu.container_module} {
  // CHECK: [[ARGS_TY:%.*]] = type { i32, i32 }
  // CHECK-DAG: @kernel_module_binary = internal constant [4 x i8] c"BLOB", align 8
  // CHECK-DAG: kernel_module_module = internal global ptr null
  // CHECK-DAG: @llvm.global_ctors = appending global {{.*}} @kernel_module_load
  // CHECK-DAG: @llvm.global_dtors = appending global {{.*}} @kernel_module_unload
  // CHECK-DAG: @kernel_module_kernel_name = private unnamed_addr constant [7 x i8] c"kernel\00", align 1
  gpu.binary @kernel_module  [#gpu.object<#nvvm.target, "BLOB">]
  llvm.func @foo() {
    // CHECK: [[ARGS:%.*]] = alloca %{{.*}}, align 8
    // CHECK: [[ARGS_ARRAY:%.*]] = alloca ptr, i64 2, align 8
    // CHECK: [[ARG0:%.*]] = getelementptr inbounds nuw [[ARGS_TY]], ptr [[ARGS]], i32 0, i32 0
    // CHECK: store i32 32, ptr [[ARG0]], align 4
    // CHECK: %{{.*}} = getelementptr ptr, ptr [[ARGS_ARRAY]], i32 0
    // CHECK: store ptr [[ARG0]], ptr %{{.*}}, align 8
    // CHECK: [[ARG1:%.*]] = getelementptr inbounds nuw [[ARGS_TY]], ptr [[ARGS]], i32 0, i32 1
    // CHECK: store i32 32, ptr [[ARG1]], align 4
    // CHECK: %{{.*}} = getelementptr ptr, ptr [[ARGS_ARRAY]], i32 1
    // CHECK: store ptr [[ARG1]], ptr %{{.*}}, align 8
    // CHECK: [[MODULE:%.*]] = load ptr, ptr @kernel_module_module
    // CHECK: [[FUNC:%.*]] = call ptr @mgpuModuleGetFunction(ptr [[MODULE]], ptr @kernel_module_kernel_name)
    // CHECK: [[STREAM:%.*]] = call ptr @mgpuStreamCreate()
    // CHECK: call void @mgpuLaunchKernel(ptr [[FUNC]], i64 8, i64 8, i64 8, i64 8, i64 8, i64 8, i32 256, ptr [[STREAM]], ptr [[ARGS_ARRAY]], ptr null, i64 2)
    // CHECK: call void @mgpuStreamSynchronize(ptr [[STREAM]])
    // CHECK: call void @mgpuStreamDestroy(ptr [[STREAM]])
    %0 = llvm.mlir.constant(8 : index) : i64
    %1 = llvm.mlir.constant(32 : i32) : i32
    %2 = llvm.mlir.constant(256 : i32) : i32
    gpu.launch_func @kernel_module::@kernel blocks in (%0, %0, %0) threads in (%0, %0, %0) : i64 dynamic_shared_memory_size %2 args(%1 : i32, %1 : i32)
    llvm.return
  }
  // CHECK: @kernel_module_load() section ".text.startup"
  // CHECK: [[MODULE:%.*]] = call ptr @mgpuModuleLoad
  // CHECK: store ptr [[MODULE]], ptr @kernel_module_module
  //
  // CHECK: @kernel_module_unload() section ".text.startup"
  // CHECK: [[MODULE:%.*]] = load ptr, ptr @kernel_module_module
  // CHECK: call void @mgpuModuleUnload(ptr [[MODULE]])
}

// -----

// Checking the correct selection of the second object using an index as a selector.
module {
  // CHECK: @kernel_module_binary = internal constant [1 x i8] c"1", align 8
  gpu.binary @kernel_module <#gpu.select_object<1>> [#gpu.object<#nvvm.target, "0">, #gpu.object<#nvvm.target, "1">]
}

// -----

// Checking the correct selection of the second object using a target as a selector.
module {
  // CHECK: @kernel_module_binary = internal constant [6 x i8] c"AMDGPU", align 8
  gpu.binary @kernel_module <#gpu.select_object<#rocdl.target>> [#gpu.object<#nvvm.target, "NVPTX">, #gpu.object<#rocdl.target, "AMDGPU">]
}

// -----

// Checking the correct selection of the second object using a target as a selector.
module {
  // CHECK: @kernel_module_binary = internal constant [4 x i8] c"BLOB", align 8
  gpu.binary @kernel_module <#gpu.select_object<#spirv.target_env<#spirv.vce<v1.0, [Addresses, Int64, Kernel], []>, api=OpenCL, #spirv.resource_limits<>>>> [#gpu.object<#nvvm.target, "NVPTX">, #gpu.object<#spirv.target_env<#spirv.vce<v1.0, [Addresses, Int64, Kernel], []>, api=OpenCL, #spirv.resource_limits<>>, "BLOB">]
}

// -----
// Checking the translation of `gpu.launch_fun` with an async dependency.
module attributes {gpu.container_module} {
  gpu.binary @kernel_module  [#gpu.object<#rocdl.target, "BLOB">]
  llvm.func @foo(%stream : !llvm.ptr) {
    %0 = llvm.mlir.constant(8 : index) : i64
    // CHECK-NOT: @mgpuStreamCreate
    // CHECK: call void @mgpuLaunchKernel
    gpu.launch_func <%stream : !llvm.ptr> @kernel_module::@kernel blocks in (%0, %0, %0) threads in (%0, %0, %0) : i64
    // CHECK-NOT: @mgpuStreamSynchronize
    // CHECK-NOT: @mgpuStreamDestroy
    llvm.return
  }
}

// -----

// Test cluster/block/thread syntax.
module attributes {gpu.container_module} {
  gpu.binary @kernel_module  [#gpu.object<#nvvm.target, "BLOB">]
  llvm.func @foo() {
  // CHECK: call void @mgpuLaunchClusterKernel(
  // CHECK-SAME: i64 1, i64 1, i64 1,
  // CHECK-SAME: i64 2, i64 2, i64 2,
  // CHECK-SAME: i64 3, i64 3, i64 3, i32 0, ptr
    %c1 = llvm.mlir.constant(1 : index) : i64
    %c2 = llvm.mlir.constant(2 : index) : i64
    %c3 = llvm.mlir.constant(3 : index) : i64
    gpu.launch_func @kernel_module::@kernel 
        clusters in (%c1, %c1, %c1)
        blocks in (%c2, %c2, %c2)
        threads in (%c3, %c3, %c3) : i64
    llvm.return
  }
}

// -----

// Checking that ELF section is populated
module attributes {gpu.container_module} {
  // CHECK: @cuda_device_mod_binary = internal constant [4 x i8] c"BLOB", section "__nv_rel_fatbin", align 8
  gpu.binary @cuda_device_mod  [#gpu.object<#nvvm.target, properties = {section = "__nv_rel_fatbin"}, "BLOB">]
}
