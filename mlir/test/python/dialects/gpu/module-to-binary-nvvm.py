# REQUIRES: host-supports-nvptx
# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
import mlir.dialects.gpu as gpu
import mlir.dialects.gpu.passes
from mlir.passmanager import *


def run(f):
    print("\nTEST:", f.__name__)
    with Context(), Location.unknown():
        f()
    return f


# CHECK-LABEL: testGPUToLLVMBin
@run
def testGPUToLLVMBin():
    with Context():
        module = Module.parse(
            r"""
module attributes {gpu.container_module} {
  gpu.module @kernel_module1 [#nvvm.target<chip = "sm_70">] {
    llvm.func @kernel(%arg0: i32, %arg1: !llvm.ptr,
        %arg2: !llvm.ptr, %arg3: i64, %arg4: i64,
        %arg5: i64) attributes {gpu.kernel} {
      llvm.return
    }
  }
}
    """
        )
    pm = PassManager("any")
    pm.add("gpu-module-to-binary{format=llvm}")
    pm.run(module.operation)
    print(module)
    # CHECK-LABEL:gpu.binary @kernel_module1
    # CHECK:[#gpu.object<#nvvm.target<chip = "sm_70">, offload = "{{.*}}">]


# CHECK-LABEL: testGPUToASMBin
@run
def testGPUToASMBin():
    with Context():
        module = Module.parse(
            r"""
module attributes {gpu.container_module} {
  gpu.module @kernel_module2 [#nvvm.target<flags = {fast}>, #nvvm.target] {
    llvm.func @kernel(%arg0: i32, %arg1: !llvm.ptr,
        %arg2: !llvm.ptr, %arg3: i64, %arg4: i64,
        %arg5: i64) attributes {gpu.kernel} {
      llvm.return
    }
  }
}
    """
        )
    pm = PassManager("any")
    pm.add("gpu-module-to-binary{format=isa}")
    pm.run(module.operation)
    print(module)
    # CHECK-LABEL:gpu.binary @kernel_module2
    # CHECK:[#gpu.object<#nvvm.target<flags = {fast}>, properties = {O = 2 : i32}, assembly = "{{.*}}">, #gpu.object<#nvvm.target, properties = {O = 2 : i32}, assembly = "{{.*}}">]
