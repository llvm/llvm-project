// RUN: mlir-opt %s --gpu-module-to-binary --verify-diagnostics --split-input-file

module attributes {gpu.container_module} {
  // expected-error @below {{the module has no target attributes}}
  gpu.module @kernel_module1 {
    llvm.func @kernel(%arg0: i32, %arg1: !llvm.ptr,
        %arg2: !llvm.ptr, %arg3: i64, %arg4: i64,
        %arg5: i64) attributes {gpu.kernel} {
      llvm.return
    }
  }
}

// -----

module attributes {gpu.container_module} {
  // expected-error @below {{An error happened while serializing the module}}
  gpu.module @kernel_module_nvvm_rocdl_op [#nvvm.target] {
    llvm.func @kernel() attributes {gpu.kernel} {
      // expected-error @below {{'rocdl.workitem.id.x' from 'rocdl' dialect is not compatible with the NVVM target}}
      %tx = rocdl.workitem.id.x : i32
      llvm.return
    }
  }
}

// -----

module attributes {gpu.container_module} {
  // expected-error @below {{An error happened while serializing the module}}
  gpu.module @kernel_module_nvvm_rocdl_barrier [#nvvm.target] {
    llvm.func @kernel() attributes {gpu.kernel} {
      // expected-error @below {{'rocdl.barrier' from 'rocdl' dialect is not compatible with the NVVM target}}
      rocdl.barrier
      llvm.return
    }
  }
}