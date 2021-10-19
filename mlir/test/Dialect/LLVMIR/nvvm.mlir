// RUN: mlir-opt %s -split-input-file -verify-diagnostics | FileCheck %s

func @nvvm_special_regs() -> i32 {
  // CHECK: nvvm.read.ptx.sreg.tid.x : i32
  %0 = nvvm.read.ptx.sreg.tid.x : i32
  // CHECK: nvvm.read.ptx.sreg.tid.y : i32
  %1 = nvvm.read.ptx.sreg.tid.y : i32
  // CHECK: nvvm.read.ptx.sreg.tid.z : i32
  %2 = nvvm.read.ptx.sreg.tid.z : i32
  // CHECK: nvvm.read.ptx.sreg.ntid.x : i32
  %3 = nvvm.read.ptx.sreg.ntid.x : i32
  // CHECK: nvvm.read.ptx.sreg.ntid.y : i32
  %4 = nvvm.read.ptx.sreg.ntid.y : i32
  // CHECK: nvvm.read.ptx.sreg.ntid.z : i32
  %5 = nvvm.read.ptx.sreg.ntid.z : i32
  // CHECK: nvvm.read.ptx.sreg.ctaid.x : i32
  %6 = nvvm.read.ptx.sreg.ctaid.x : i32
  // CHECK: nvvm.read.ptx.sreg.ctaid.y : i32
  %7 = nvvm.read.ptx.sreg.ctaid.y : i32
  // CHECK: nvvm.read.ptx.sreg.ctaid.z : i32
  %8 = nvvm.read.ptx.sreg.ctaid.z : i32
  // CHECK: nvvm.read.ptx.sreg.nctaid.x : i32
  %9 = nvvm.read.ptx.sreg.nctaid.x : i32
  // CHECK: nvvm.read.ptx.sreg.nctaid.y : i32
  %10 = nvvm.read.ptx.sreg.nctaid.y : i32
  // CHECK: nvvm.read.ptx.sreg.nctaid.z : i32
  %11 = nvvm.read.ptx.sreg.nctaid.z : i32
  llvm.return %0 : i32
}

func @llvm.nvvm.barrier0() {
  // CHECK: nvvm.barrier0
  nvvm.barrier0
  llvm.return
}

func @nvvm_shfl(
    %arg0 : i32, %arg1 : i32, %arg2 : i32,
    %arg3 : i32, %arg4 : f32) -> i32 {
  // CHECK: nvvm.shfl.sync.bfly %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : i32
  %0 = nvvm.shfl.sync.bfly %arg0, %arg3, %arg1, %arg2 : i32
  // CHECK: nvvm.shfl.sync.bfly %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : f32
  %1 = nvvm.shfl.sync.bfly %arg0, %arg4, %arg1, %arg2 : f32
  llvm.return %0 : i32
}

func @nvvm_shfl_pred(
    %arg0 : i32, %arg1 : i32, %arg2 : i32,
    %arg3 : i32, %arg4 : f32) -> !llvm.struct<(i32, i1)> {
  // CHECK: nvvm.shfl.sync.bfly %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : !llvm.struct<(i32, i1)>
  %0 = nvvm.shfl.sync.bfly %arg0, %arg3, %arg1, %arg2 {return_value_and_is_valid} : !llvm.struct<(i32, i1)>
  // CHECK: nvvm.shfl.sync.bfly %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : !llvm.struct<(f32, i1)>
  %1 = nvvm.shfl.sync.bfly %arg0, %arg4, %arg1, %arg2 {return_value_and_is_valid} : !llvm.struct<(f32, i1)>
  llvm.return %0 : !llvm.struct<(i32, i1)>
}

func @nvvm_vote(%arg0 : i32, %arg1 : i1) -> i32 {
  // CHECK: nvvm.vote.ballot.sync %{{.*}}, %{{.*}} : i32
  %0 = nvvm.vote.ballot.sync %arg0, %arg1 : i32
  llvm.return %0 : i32
}

func @nvvm_mma(%a0 : vector<2xf16>, %a1 : vector<2xf16>,
               %b0 : vector<2xf16>, %b1 : vector<2xf16>,
               %c0 : f32, %c1 : f32, %c2 : f32, %c3 : f32,
               %c4 : f32, %c5 : f32, %c6 : f32, %c7 : f32) {
  // CHECK: nvvm.mma.sync {{.*}} {alayout = "row", blayout = "col"} : (vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, f32, f32, f32, f32, f32, f32, f32, f32) -> !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
  %0 = nvvm.mma.sync %a0, %a1, %b0, %b1, %c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7 {alayout="row", blayout="col"} : (vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, f32, f32, f32, f32, f32, f32, f32, f32) -> !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
  llvm.return %0 : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
}

// -----

// expected-error@below {{attribute attached to unexpected op}}
func private @expected_llvm_func() attributes { nvvm.kernel }
