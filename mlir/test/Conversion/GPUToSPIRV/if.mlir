// RUN: mlir-opt -convert-gpu-to-spirv %s -o - | FileCheck %s

module attributes {gpu.container_module} {
  func @main(%arg0 : memref<10xf32>, %arg1 : i1) {
    %c0 = constant 1 : index
    "gpu.launch_func"(%c0, %c0, %c0, %c0, %c0, %c0, %arg0, %arg1) { kernel = "kernel_simple_selection", kernel_module = @kernels} : (index, index, index, index, index, index, memref<10xf32>, i1) -> ()
    return
  }

  gpu.module @kernels {
    // CHECK-LABEL: @kernel_simple_selection
    gpu.func @kernel_simple_selection(%arg2 : memref<10xf32>, %arg3 : i1)
    attributes {gpu.kernel} {
      %value = constant 0.0 : f32
      %i = constant 0 : index

      // CHECK:       spv.selection {
      // CHECK-NEXT:    spv.BranchConditional {{%.*}}, [[TRUE:\^.*]], [[MERGE:\^.*]]
      // CHECK-NEXT:  [[TRUE]]:
      // CHECK:         spv.Branch [[MERGE]]
      // CHECK-NEXT:  [[MERGE]]:
      // CHECK-NEXT:    spv._merge
      // CHECK-NEXT:  }
      // CHECK-NEXT:  spv.Return

      loop.if %arg3 {
        store %value, %arg2[%i] : memref<10xf32>
      }
      gpu.return
    }

    // CHECK-LABEL: @kernel_nested_selection
    gpu.func @kernel_nested_selection(%arg3 : memref<10xf32>, %arg4 : memref<10xf32>, %arg5 : i1, %arg6 : i1)
    attributes {gpu.kernel} {
      %i = constant 0 : index
      %j = constant 9 : index

      // CHECK:       spv.selection {
      // CHECK-NEXT:    spv.BranchConditional {{%.*}}, [[TRUE_TOP:\^.*]], [[FALSE_TOP:\^.*]]
      // CHECK-NEXT:  [[TRUE_TOP]]:
      // CHECK-NEXT:    spv.selection {
      // CHECK-NEXT:      spv.BranchConditional {{%.*}}, [[TRUE_NESTED_TRUE_PATH:\^.*]], [[FALSE_NESTED_TRUE_PATH:\^.*]]
      // CHECK-NEXT:    [[TRUE_NESTED_TRUE_PATH]]:
      // CHECK:           spv.Branch [[MERGE_NESTED_TRUE_PATH:\^.*]]
      // CHECK-NEXT:    [[FALSE_NESTED_TRUE_PATH]]:
      // CHECK:           spv.Branch [[MERGE_NESTED_TRUE_PATH]]
      // CHECK-NEXT:    [[MERGE_NESTED_TRUE_PATH]]:
      // CHECK-NEXT:      spv._merge
      // CHECK-NEXT:    }
      // CHECK-NEXT:    spv.Branch [[MERGE_TOP:\^.*]]
      // CHECK-NEXT:  [[FALSE_TOP]]:
      // CHECK-NEXT:    spv.selection {
      // CHECK-NEXT:      spv.BranchConditional {{%.*}}, [[TRUE_NESTED_FALSE_PATH:\^.*]], [[FALSE_NESTED_FALSE_PATH:\^.*]]
      // CHECK-NEXT:    [[TRUE_NESTED_FALSE_PATH]]:
      // CHECK:           spv.Branch [[MERGE_NESTED_FALSE_PATH:\^.*]]
      // CHECK-NEXT:    [[FALSE_NESTED_FALSE_PATH]]:
      // CHECK:           spv.Branch [[MERGE_NESTED_FALSE_PATH]]
      // CHECK:         [[MERGE_NESTED_FALSE_PATH]]:
      // CHECK-NEXT:      spv._merge
      // CHECK-NEXT:    }
      // CHECK-NEXT:    spv.Branch [[MERGE_TOP]]
      // CHECK-NEXT:  [[MERGE_TOP]]:
      // CHECK-NEXT:    spv._merge
      // CHECK-NEXT:  }
      // CHECK-NEXT:  spv.Return

      loop.if %arg5 {
        loop.if %arg6 {
          %value = load %arg3[%i] : memref<10xf32>
          store %value, %arg4[%i] : memref<10xf32>
        } else {
          %value = load %arg4[%i] : memref<10xf32>
          store %value, %arg3[%i] : memref<10xf32>
        }
      } else {
        loop.if %arg6 {
          %value = load %arg3[%j] : memref<10xf32>
          store %value, %arg4[%j] : memref<10xf32>
        } else {
          %value = load %arg4[%j] : memref<10xf32>
          store %value, %arg3[%j] : memref<10xf32>
        }
      }
      gpu.return
    }
  }
}
