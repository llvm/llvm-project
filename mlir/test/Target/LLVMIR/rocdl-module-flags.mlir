// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

module {
  llvm.module_flags [
    #rocdl.buffer_oob_mode_flag<any>,
    #rocdl.tbuffer_oob_mode_flag<relaxed>
  ]
  llvm.func @oob_any_relaxed() {
    llvm.return
  }
}

// CHECK-LABEL: define void @oob_any_relaxed()
// CHECK: !llvm.module.flags = !{![[BUFFER_ANY:[0-9]+]], ![[TBUFFER_RELAXED:[0-9]+]]
// CHECK-DAG: ![[BUFFER_ANY]] = !{i32 7, !"amdgpu.buffer.oob.mode", i32 0}
// CHECK-DAG: ![[TBUFFER_RELAXED]] = !{i32 7, !"amdgpu.tbuffer.oob.mode", i32 1}

// -----

module {
  llvm.module_flags [
    #rocdl.buffer_oob_mode_flag<strict>
  ]
  llvm.func @oob_strict() {
    llvm.return
  }
}

// CHECK-LABEL: define void @oob_strict()
// CHECK: !llvm.module.flags = !{![[BUFFER_STRICT:[0-9]+]]
// CHECK-DAG: ![[BUFFER_STRICT]] = !{i32 7, !"amdgpu.buffer.oob.mode", i32 2}

// -----

module {
  llvm.module_flags [
    #llvm.mlir.module_flag<max, "amdgpu.buffer.oob.mode",
                           #rocdl.buffer_oob_mode<relaxed>>
  ]
  llvm.func @generic_oob_relaxed() {
    llvm.return
  }
}

// CHECK-LABEL: define void @generic_oob_relaxed()
// CHECK: !llvm.module.flags = !{![[GENERIC_BUFFER_RELAXED:[0-9]+]]
// CHECK-DAG: ![[GENERIC_BUFFER_RELAXED]] = !{i32 7, !"amdgpu.buffer.oob.mode", i32 1}
