// RUN: mlir-opt %s -cse -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: @nvvm_special_regs_clock
llvm.func @nvvm_special_regs_clock() -> !llvm.struct<(i32, i32)> {
  %0 = llvm.mlir.zero: !llvm.struct<(i32, i32)>
  // CHECK:  {{.*}} = nvvm.read.ptx.sreg.clock
  %1 = nvvm.read.ptx.sreg.clock : i32
  // CHECK:  {{.*}} = nvvm.read.ptx.sreg.clock
  %2 = nvvm.read.ptx.sreg.clock : i32
  %4 = llvm.insertvalue %1, %0[0]: !llvm.struct<(i32, i32)>
  %5 = llvm.insertvalue %2, %4[1]: !llvm.struct<(i32, i32)>
  llvm.return %5: !llvm.struct<(i32, i32)>
}

// CHECK-LABEL: @nvvm_special_regs_clock64
llvm.func @nvvm_special_regs_clock64() -> !llvm.struct<(i64, i64)> {
  %0 = llvm.mlir.zero: !llvm.struct<(i64, i64)>
  // CHECK:  {{.*}} = nvvm.read.ptx.sreg.clock64
  %1 = nvvm.read.ptx.sreg.clock64 : i64
  // CHECK:  {{.*}} = nvvm.read.ptx.sreg.clock64
  %2 = nvvm.read.ptx.sreg.clock64 : i64
  %4 = llvm.insertvalue %1, %0[0]: !llvm.struct<(i64, i64)>
  %5 = llvm.insertvalue %2, %4[1]: !llvm.struct<(i64, i64)>
  llvm.return %5: !llvm.struct<(i64, i64)>
}

// CHECK-LABEL: @nvvm_special_regs_globaltimer
llvm.func @nvvm_special_regs_globaltimer() -> !llvm.struct<(i64, i64)> {
  %0 = llvm.mlir.zero: !llvm.struct<(i64, i64)>
  // CHECK:  {{.*}} = nvvm.read.ptx.sreg.globaltimer
  %1 = nvvm.read.ptx.sreg.globaltimer : i64
  // CHECK:  {{.*}} = nvvm.read.ptx.sreg.globaltimer
  %2 = nvvm.read.ptx.sreg.globaltimer : i64
  %4 = llvm.insertvalue %1, %0[0]: !llvm.struct<(i64, i64)>
  %5 = llvm.insertvalue %2, %4[1]: !llvm.struct<(i64, i64)>
  llvm.return %5: !llvm.struct<(i64, i64)>
}
