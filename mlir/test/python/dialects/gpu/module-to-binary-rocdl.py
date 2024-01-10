# REQUIRES: host-supports-amdgpu
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
  gpu.module @kernel_module1 [#rocdl.target<chip = "gfx90a">] {
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
    # CHECK:[#gpu.object<#rocdl.target<chip = "gfx90a">, offload = "{{.*}}">]


# CHECK-LABEL: testGPUToASMBin
@run
def testGPUToASMBin():
    with Context():
        module = Module.parse(
            r"""
module attributes {gpu.container_module} {
  gpu.module @kernel_module2 [#rocdl.target<flags = {fast}>, #rocdl.target] {
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
    # CHECK:[#gpu.object<#rocdl.target<flags = {fast}>, assembly = "{{.*}}">, #gpu.object<#rocdl.target, assembly = "{{.*}}">]
