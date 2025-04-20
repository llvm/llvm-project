// RUN: mlir-opt -int-range-optimizations %s | FileCheck %s
gpu.module @module{
    gpu.func @kernel_1() kernel {
        %tidx = nvvm.read.ptx.sreg.tid.x range <i32, 0, 32> : i32
        %tidy = nvvm.read.ptx.sreg.tid.y range <i32, 0, 128> : i32
        %tidz = nvvm.read.ptx.sreg.tid.z range <i32, 0, 4> : i32
        %c64 = arith.constant 64 : i32
        
        %1 = arith.cmpi sgt, %tidx, %c64 : i32
        scf.if %1 {
            gpu.printf "threadidx"
        }
        %2 = arith.cmpi sgt, %tidy, %c64 : i32
        scf.if %2 {
            gpu.printf "threadidy"
        }
        %3 = arith.cmpi sgt, %tidz, %c64 : i32
        scf.if %3 {
            gpu.printf "threadidz"
        }
        gpu.return
    }
}

// CHECK-LABEL: gpu.func @kernel_1
// CHECK: %[[false:.+]] = arith.constant false
// CHECK: %[[c64_i32:.+]] = arith.constant 64 : i32
// CHECK: %[[S0:.+]] = nvvm.read.ptx.sreg.tid.y range <i32, 0, 128> : i32
// CHECK: scf.if %[[false]] {
// CHECK: gpu.printf "threadidx"
// CHECK: %[[S1:.+]] = arith.cmpi sgt, %[[S0]], %[[c64_i32]] : i32
// CHECK: scf.if %[[S1]] {
// CHECK: gpu.printf "threadidy"
// CHECK: scf.if %[[false]] {
// CHECK: gpu.printf "threadidz"
