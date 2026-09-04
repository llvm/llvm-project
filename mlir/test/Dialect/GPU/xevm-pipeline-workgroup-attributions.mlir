// RUN: mlir-opt %s \
// RUN:   --convert-gpu-to-llvm-spv="use-64bit-index=true \
// RUN:     encode-workgroup-attributions-as-arguments=true" \
// RUN:   | FileCheck %s --check-prefix=ARGS
// RUN: mlir-opt %s \
// RUN:   --gpu-lower-to-xevm-pipeline="xegpu-op-level=lane binary-format=llvm" \
// RUN:   --mlir-print-ir-after=convert-gpu-to-llvm-spv -o /dev/null 2>&1 \
// RUN:   | FileCheck %s --check-prefix=XEVM

// Static workgroup attributions are not launch operands. Check that the XeVM
// pipeline materializes them as workgroup address-space globals instead of
// adding hidden kernel arguments that the host launch does not initialize.

module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @kernel()
        workgroup(%scratch: memref<8xf32, 3>) kernel {
      %c0 = arith.constant 0 : index
      %one = arith.constant 1.0 : f32
      memref.store %one, %scratch[%c0] : memref<8xf32, 3>
      gpu.return
    }
  }
}

// With argument encoding enabled, the workgroup attribution becomes a hidden
// local-pointer kernel argument.
// ARGS-NOT: llvm.mlir.global internal @__wg_kernel_0
// ARGS-LABEL: llvm.func spir_kernelcc @kernel(
// ARGS-SAME: %[[SCRATCH:[a-zA-Z0-9_]+]]: !llvm.ptr<3>
// ARGS-SAME: llvm.workgroup_attribution = #llvm.mlir.workgroup_attribution<8 : i64, f32>
// ARGS: llvm.insertvalue %[[SCRATCH]],

// The XeVM pipeline disables argument encoding and materializes the static
// attribution as a workgroup address-space global instead.
// XEVM: convert-gpu-to-llvm-spv{encode-workgroup-attributions-as-arguments=false
// XEVM: llvm.mlir.global internal @__wg_kernel_0()
// XEVM-SAME: {addr_space = 3 : i32} : !llvm.array<8 x f32>
// XEVM-LABEL: llvm.func spir_kernelcc @kernel() attributes {gpu.kernel}
// XEVM: llvm.mlir.addressof @__wg_kernel_0 : !llvm.ptr<3>
