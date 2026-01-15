// RUN: mlir-opt %s -split-input-file -convert-gpu-to-rocdl | FileCheck %s --check-prefixes=ROCDL
// RUN: mlir-opt %s -split-input-file -convert-gpu-to-nvvm | FileCheck %s --check-prefixes=NVVM

gpu.module @kernel {
  gpu.func @gpu_global_id() -> (index) {
    %global_id_x = gpu.global_id x
    gpu.return %global_id_x : index
  }
}

// ROCDL-LABEL:   llvm.func @gpu_global_id() -> i64 {
// ROCDL:           %[[WORKGROUP_0:.*]] = rocdl.workgroup.id.x : i32
// ROCDL:           %[[SEXT_0:.*]] = llvm.sext %[[WORKGROUP_0]] : i32 to i64
// ROCDL:           %[[WORKGROUP_1:.*]] = rocdl.workgroup.dim.x : i32
// ROCDL:           %[[SEXT_1:.*]] = llvm.sext %[[WORKGROUP_1]] : i32 to i64
// ROCDL:           %[[MUL_0:.*]] = llvm.mul %[[SEXT_0]], %[[SEXT_1]] : i64
// ROCDL:           %[[WORKITEM_0:.*]] = rocdl.workitem.id.x : i32
// ROCDL:           %[[SEXT_2:.*]] = llvm.sext %[[WORKITEM_0]] : i32 to i64
// ROCDL:           %[[ADD_0:.*]] = llvm.add %[[SEXT_2]], %[[MUL_0]] : i64
// ROCDL:           llvm.return %[[ADD_0]] : i64
// ROCDL:         }

// NVVM-LABEL:   llvm.func @gpu_global_id() -> i64 {
// NVVM:           %[[READ_0:.*]] = nvvm.read.ptx.sreg.ctaid.x : i32
// NVVM:           %[[SEXT_0:.*]] = llvm.sext %[[READ_0]] : i32 to i64
// NVVM:           %[[READ_1:.*]] = nvvm.read.ptx.sreg.ntid.x : i32
// NVVM:           %[[SEXT_1:.*]] = llvm.sext %[[READ_1]] : i32 to i64
// NVVM:           %[[MUL_0:.*]] = llvm.mul %[[SEXT_0]], %[[SEXT_1]] : i64
// NVVM:           %[[READ_2:.*]] = nvvm.read.ptx.sreg.tid.x : i32
// NVVM:           %[[SEXT_2:.*]] = llvm.sext %[[READ_2]] : i32 to i64
// NVVM:           %[[ADD_0:.*]] = llvm.add %[[SEXT_2]], %[[MUL_0]] : i64
// NVVM:           llvm.return %[[ADD_0]] : i64
// NVVM:         }
