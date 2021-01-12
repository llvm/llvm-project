// RUN: mlir-translate -mlir-to-nvvmir %s | FileCheck %s

llvm.func @nvvm_special_regs() -> i32 {
  // CHECK: %1 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %1 = nvvm.read.ptx.sreg.tid.x : i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %2 = nvvm.read.ptx.sreg.tid.y : i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.tid.z()
  %3 = nvvm.read.ptx.sreg.tid.z : i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %4 = nvvm.read.ptx.sreg.ntid.x : i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
  %5 = nvvm.read.ptx.sreg.ntid.y : i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.ntid.z()
  %6 = nvvm.read.ptx.sreg.ntid.z : i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %7 = nvvm.read.ptx.sreg.ctaid.x : i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %8 = nvvm.read.ptx.sreg.ctaid.y : i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.ctaid.z()
  %9 = nvvm.read.ptx.sreg.ctaid.z : i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
  %10 = nvvm.read.ptx.sreg.nctaid.x : i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y()
  %11 = nvvm.read.ptx.sreg.nctaid.y : i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z()
  %12 = nvvm.read.ptx.sreg.nctaid.z : i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.warpsize()
  %13 = nvvm.read.ptx.sreg.warpsize : i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.laneid()
  %14 = nvvm.read.ptx.sreg.laneid : i32
  llvm.return %1 : i32
}

llvm.func @llvm.nvvm.barrier0() {
  // CHECK: call void @llvm.nvvm.barrier0()
  nvvm.barrier0
  llvm.return
}

llvm.func @nvvm_shfl(
    %0 : i32, %1 : i32, %2 : i32,
    %3 : i32, %4 : f32) -> i32 {
  // CHECK: call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %6 = nvvm.shfl.sync.bfly %0, %3, %1, %2 : i32
  // CHECK: call float @llvm.nvvm.shfl.sync.bfly.f32(i32 %{{.*}}, float %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %7 = nvvm.shfl.sync.bfly %0, %4, %1, %2 : f32
  llvm.return %6 : i32
}

llvm.func @nvvm_shfl_pred(
    %0 : i32, %1 : i32, %2 : i32,
    %3 : i32, %4 : f32) -> !llvm.struct<(i32, i1)> {
  // CHECK: call { i32, i1 } @llvm.nvvm.shfl.sync.bfly.i32p(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %6 = nvvm.shfl.sync.bfly %0, %3, %1, %2 {return_value_and_is_valid} : !llvm.struct<(i32, i1)>
  // CHECK: call { float, i1 } @llvm.nvvm.shfl.sync.bfly.f32p(i32 %{{.*}}, float %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %7 = nvvm.shfl.sync.bfly %0, %4, %1, %2 {return_value_and_is_valid} : !llvm.struct<(f32, i1)>
  llvm.return %6 : !llvm.struct<(i32, i1)>
}

llvm.func @nvvm_vote(%0 : i32, %1 : i1) -> i32 {
  // CHECK: call i32 @llvm.nvvm.vote.ballot.sync(i32 %{{.*}}, i1 %{{.*}})
  %3 = nvvm.vote.ballot.sync %0, %1 : i32
  llvm.return %3 : i32
}

llvm.func @nvvm_mma(%a0 : vector<2xf16>, %a1 : vector<2xf16>,
                    %b0 : vector<2xf16>, %b1 : vector<2xf16>,
                    %c0 : f32, %c1 : f32, %c2 : f32, %c3 : f32,
                    %c4 : f32, %c5 : f32, %c6 : f32, %c7 : f32) {
  // CHECK: call { float, float, float, float, float, float, float, float } @llvm.nvvm.mma.m8n8k4.row.col.f32.f32
  %0 = nvvm.mma.sync %a0, %a1, %b0, %b1, %c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7 {alayout="row", blayout="col"} : (vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, f32, f32, f32, f32, f32, f32, f32, f32) -> !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
  llvm.return %0 : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
}

// This function has the "kernel" attribute attached and should appear in the
// NVVM annotations after conversion.
llvm.func @kernel_func() attributes {gpu.kernel} {
  llvm.return
}

// CHECK:     !nvvm.annotations =
// CHECK-NOT: {i32 ()* @nvvm_special_regs, !"kernel", i32 1}
// CHECK:     {void ()* @kernel_func, !"kernel", i32 1}
