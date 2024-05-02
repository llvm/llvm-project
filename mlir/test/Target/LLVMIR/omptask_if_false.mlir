// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", omp.is_gpu = false, omp.is_target_device = false, omp.version = #omp.version<version = 11>}
{
  llvm.func @foo_(%arg0: !llvm.ptr {fir.bindc_name = "n"}, %arg1: !llvm.ptr {fir.bindc_name = "r"}) attributes {fir.internal_name = "_QPfoo"} {
    %0 = llvm.mlir.constant(false) : i1
    omp.task if(%0) depend(taskdependin -> %arg0 : !llvm.ptr) {
      %1 = llvm.load %arg0 : !llvm.ptr -> i32
      llvm.store %1, %arg1 : i32, !llvm.ptr
      omp.terminator
    }
    llvm.return
  }
  llvm.func @llvm.stacksave.p0() -> !llvm.ptr attributes {sym_visibility = "private"}
  llvm.func @llvm.stackrestore.p0(!llvm.ptr) attributes {sym_visibility = "private"}
}


// CHECK: call void @__kmpc_omp_taskwait_deps_51
// CHECK-NEXT: call void @__kmpc_omp_task_begin_if0
// CHECK-NEXT: call void @foo_..omp_par
// CHECK-NEXT: call void @__kmpc_omp_task_complete_if0

