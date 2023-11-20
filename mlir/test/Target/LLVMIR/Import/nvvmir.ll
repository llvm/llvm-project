; RUN: mlir-translate -import-llvm %s | FileCheck %s

define i32 @nvvm_special_regs() {
  %1 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.tid.z()
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
  %6 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.z()
  %7 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %8 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %9 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.z()
  %10 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
  %11 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y()
  %12 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z()
  %13 = call i32 @llvm.nvvm.read.ptx.sreg.warpsize()
  %14 = call i32 @llvm.nvvm.read.ptx.sreg.laneid()
  %15 = call i32 @llvm.nvvm.read.ptx.sreg.clusterid.x()
  %16 = call i32 @llvm.nvvm.read.ptx.sreg.clusterid.y()
  %17 = call i32 @llvm.nvvm.read.ptx.sreg.clusterid.z()
  %18 = call i32 @llvm.nvvm.read.ptx.sreg.nclusterid.x()
  %19 = call i32 @llvm.nvvm.read.ptx.sreg.nclusterid.y()
  %20 = call i32 @llvm.nvvm.read.ptx.sreg.nclusterid.z()
  %21 = call i32 @llvm.nvvm.read.ptx.sreg.cluster.ctaid.x()
  %22 = call i32 @llvm.nvvm.read.ptx.sreg.cluster.ctaid.y()
  %23 = call i32 @llvm.nvvm.read.ptx.sreg.cluster.ctaid.z()
  %24 = call i32 @llvm.nvvm.read.ptx.sreg.cluster.nctaid.x()
  %25 = call i32 @llvm.nvvm.read.ptx.sreg.cluster.nctaid.y()
  %26 = call i32 @llvm.nvvm.read.ptx.sreg.cluster.nctaid.z()
  %27 = call i32 @llvm.nvvm.read.ptx.sreg.cluster.ctarank()
  %28 = call i32 @llvm.nvvm.read.ptx.sreg.cluster.nctarank()
  ret i32 %1
}

define float @nvvm_rcp(float %0) {
  %2 = call float @llvm.nvvm.rcp.approx.ftz.f(float %0)
  ret float %2
}

define void @llvm_nvvm_barrier0() {
  call void @llvm.nvvm.barrier0()
  ret void
}

define i32 @nvvm_shfl(i32 %0, i32 %1, i32 %2, i32 %3, float %4) {
  %6 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 %0, i32 %3, i32 %1, i32 %2)
  %7 = call float @llvm.nvvm.shfl.sync.bfly.f32(i32 %0, float %4, i32 %1, i32 %2)
  %8 = call i32 @llvm.nvvm.shfl.sync.up.i32(i32 %0, i32 %3, i32 %1, i32 %2)
  %9 = call float @llvm.nvvm.shfl.sync.up.f32(i32 %0, float %4, i32 %1, i32 %2)
  %10 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 %0, i32 %3, i32 %1, i32 %2)
  %11 = call float @llvm.nvvm.shfl.sync.down.f32(i32 %0, float %4, i32 %1, i32 %2)
  %12 = call i32 @llvm.nvvm.shfl.sync.idx.i32(i32 %0, i32 %3, i32 %1, i32 %2)
  %13 = call float @llvm.nvvm.shfl.sync.idx.f32(i32 %0, float %4, i32 %1, i32 %2)
  ret i32 %6
}

define { i32, i1 } @nvvm_shfl_pred(i32 %0, i32 %1, i32 %2, i32 %3, float %4) {
  %6 = call { i32, i1 } @llvm.nvvm.shfl.sync.bfly.i32p(i32 %0, i32 %3, i32 %1, i32 %2)
  %7 = call { float, i1 } @llvm.nvvm.shfl.sync.bfly.f32p(i32 %0, float %4, i32 %1, i32 %2)
  %8 = call { i32, i1 } @llvm.nvvm.shfl.sync.up.i32p(i32 %0, i32 %3, i32 %1, i32 %2)
  %9 = call { float, i1 } @llvm.nvvm.shfl.sync.up.f32p(i32 %0, float %4, i32 %1, i32 %2)
  %10 = call { i32, i1 } @llvm.nvvm.shfl.sync.down.i32p(i32 %0, i32 %3, i32 %1, i32 %2)
  %11 = call { float, i1 } @llvm.nvvm.shfl.sync.down.f32p(i32 %0, float %4, i32 %1, i32 %2)
  %12 = call { i32, i1 } @llvm.nvvm.shfl.sync.idx.i32p(i32 %0, i32 %3, i32 %1, i32 %2)
  %13 = call { float, i1 } @llvm.nvvm.shfl.sync.idx.f32p(i32 %0, float %4, i32 %1, i32 %2)
  ret { i32, i1 } %6
}

define i32 @nvvm_vote(i32 %0, i1 %1) {
  %3 = call i32 @llvm.nvvm.vote.ballot.sync(i32 %0, i1 %1)
  ret i32 %3
}

define { float, float, float, float, float, float, float, float } @nvvm_mma_mn8n8k4_row_col_f32_f32(<2 x half> %0, <2 x half> %1, <2 x half> %2, <2 x half> %3, float %4, float %5, float %6, float %7, float %8, float %9, float %10, float %11) {
  %13 = call { float, float, float, float, float, float, float, float } @llvm.nvvm.mma.m8n8k4.row.col.f32.f32(<2 x half> %0, <2 x half> %1, <2 x half> %2, <2 x half> %3, float %4, float %5, float %6, float %7, float %8, float %9, float %10, float %11)
  ret { float, float, float, float, float, float, float, float } %13
}

define { <2 x half>, <2 x half> } @nvvm_mma_m16n8k16_f16_f16(<2 x half> %0, <2 x half> %1, <2 x half> %2, <2 x half> %3, <2 x half> %4, <2 x half> %5, <2 x half> %6, <2 x half> %7) {
  %9 = call { <2 x half>, <2 x half> } @llvm.nvvm.mma.m16n8k16.row.col.f16.f16(<2 x half> %0, <2 x half> %1, <2 x half> %2, <2 x half> %3, <2 x half> %4, <2 x half> %5, <2 x half> %6, <2 x half> %7)
  ret { <2 x half>, <2 x half> } %9
}

define { float, float, float, float } @nvvm_mma_m16n8k16_f32_f16(<2 x half> %0, <2 x half> %1, <2 x half> %2, <2 x half> %3, <2 x half> %4, <2 x half> %5, <2 x half> %6, <2 x half> %7) {
  %9 = call { float, float, float, float } @llvm.nvvm.mma.m16n8k16.row.col.f32.f16(<2 x half> %0, <2 x half> %1, <2 x half> %2, <2 x half> %3, <2 x half> %4, <2 x half> %5, <2 x half> %6, <2 x half> %7)
  ret { float, float, float, float } %9
}

define { <2 x half>, <2 x half> } @nvvm_mma_m16n8k16_f16_f32(<2 x half> %0, <2 x half> %1, <2 x half> %2, <2 x half> %3, <2 x half> %4, <2 x half> %5, float %6, float %7, float %8, float %9) {
  %11 = call { <2 x half>, <2 x half> } @llvm.nvvm.mma.m16n8k16.row.col.f16.f32(<2 x half> %0, <2 x half> %1, <2 x half> %2, <2 x half> %3, <2 x half> %4, <2 x half> %5, float %6, float %7, float %8, float %9)
  ret { <2 x half>, <2 x half> } %11
}

define { float, float, float, float } @nvvm_mma_m16n8k16_f32_f32(<2 x half> %0, <2 x half> %1, <2 x half> %2, <2 x half> %3, <2 x half> %4, <2 x half> %5, float %6, float %7, float %8, float %9) {
  %11 = call { float, float, float, float } @llvm.nvvm.mma.m16n8k16.row.col.f32.f32(<2 x half> %0, <2 x half> %1, <2 x half> %2, <2 x half> %3, <2 x half> %4, <2 x half> %5, float %6, float %7, float %8, float %9)
  ret { float, float, float, float } %11
}

define { i32, i32, i32, i32 } @nvvm_mma_m16n8k16_s8_s8(i32 %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6) {
  %8 = call { i32, i32, i32, i32 } @llvm.nvvm.mma.m16n8k16.row.col.s8(i32 %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6)
  ret { i32, i32, i32, i32 } %8
}

define { i32, i32, i32, i32 } @nvvm_mma_m16n8k16_s8_u8(i32 %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6) {
  %8 = call { i32, i32, i32, i32 } @llvm.nvvm.mma.m16n8k16.row.col.satfinite.s8.u8(i32 %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6)
  ret { i32, i32, i32, i32 } %8
}

define { i32, i32, i32, i32 } @nvvm_mma_m16n8k128_b1_b1(i32 %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6) {
  %8 = call { i32, i32, i32, i32 } @llvm.nvvm.mma.xor.popc.m16n8k128.row.col.b1(i32 %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6)
  ret { i32, i32, i32, i32 } %8
}

define { i32, i32, i32, i32 } @nvvm_mma_m16n8k32_s4_s4(i32 %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6) {
  %8 = call { i32, i32, i32, i32 } @llvm.nvvm.mma.m16n8k32.row.col.satfinite.s4(i32 %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6)
  ret { i32, i32, i32, i32 } %8
}

define { double, double } @nvvm_mma_m8n8k4_f64_f64(double %0, double %1, double %2, double %3) {
  %5 = call { double, double } @llvm.nvvm.mma.m8n8k4.row.col.f64(double %0, double %1, double %2, double %3)
  ret { double, double } %5
}

define { float, float, float, float } @nvvm_mma_m16n8k4_tf32_f32(i32 %0, i32 %1, i32 %2, float %3, float %4, float %5, float %6) {
  %8 = call { float, float, float, float } @llvm.nvvm.mma.m16n8k4.row.col.tf32(i32 %0, i32 %1, i32 %2, float %3, float %4, float %5, float %6)
  ret { float, float, float, float } %8
}

define void @gpu_wmma_load_op(ptr addrspace(3) %0, i32 %1) {
  %3 = call { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.load.a.row.stride.f16.p3(ptr addrspace(3) %0, i32 %1)
  ret void
}

define void @gpu_wmma_store_op(ptr addrspace(3) %0, i32 %1, <2 x half> %2, <2 x half> %3, <2 x half> %4, <2 x half> %5) {
  call void @llvm.nvvm.wmma.m16n16k16.store.d.row.stride.f16.p3(ptr addrspace(3) %0, <2 x half> %2, <2 x half> %3, <2 x half> %4, <2 x half> %5, i32 %1)
  ret void
}

define void @gpu_wmma_mma_op(<2 x half> %0, <2 x half> %1, <2 x half> %2, <2 x half> %3, <2 x half> %4, <2 x half> %5, <2 x half> %6, <2 x half> %7, <2 x half> %8, <2 x half> %9, <2 x half> %10, <2 x half> %11, <2 x half> %12, <2 x half> %13, <2 x half> %14, <2 x half> %15, <2 x half> %16, <2 x half> %17, <2 x half> %18, <2 x half> %19) {
  %21 = call { <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.mma.row.row.f16.f16(<2 x half> %0, <2 x half> %1, <2 x half> %2, <2 x half> %3, <2 x half> %4, <2 x half> %5, <2 x half> %6, <2 x half> %7, <2 x half> %8, <2 x half> %9, <2 x half> %10, <2 x half> %11, <2 x half> %12, <2 x half> %13, <2 x half> %14, <2 x half> %15, <2 x half> %16, <2 x half> %17, <2 x half> %18, <2 x half> %19)
  ret void
}

define void @nvvm_wmma_load_tf32(ptr %0, i32 %1) {
  %3 = call { i32, i32, i32, i32 } @llvm.nvvm.wmma.m16n16k8.load.a.row.stride.tf32.p0(ptr %0, i32 %1)
  ret void
}

define void @nvvm_wmma_mma(i32 %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, float %8, float %9, float %10, float %11, float %12, float %13, float %14, float %15) {
  %17 = call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k8.mma.row.row.tf32(i32 %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, float %8, float %9, float %10, float %11, float %12, float %13, float %14, float %15)
  ret void
}

define void @cp_async(ptr addrspace(3) %0, ptr addrspace(1) %1) {
  call void @llvm.nvvm.cp.async.ca.shared.global.4(ptr addrspace(3) %0, ptr addrspace(1) %1)
  call void @llvm.nvvm.cp.async.ca.shared.global.8(ptr addrspace(3) %0, ptr addrspace(1) %1)
  call void @llvm.nvvm.cp.async.ca.shared.global.16(ptr addrspace(3) %0, ptr addrspace(1) %1)
  call void @llvm.nvvm.cp.async.cg.shared.global.16(ptr addrspace(3) %0, ptr addrspace(1) %1)
  call void @llvm.nvvm.cp.async.commit.group()
  call void @llvm.nvvm.cp.async.wait.group(i32 0)
  ret void
}

define void @ld_matrix(ptr addrspace(3) %0) {
  %2 = call i32 @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x1.b16.p3(ptr addrspace(3) %0)
  %3 = call { i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x2.b16.p3(ptr addrspace(3) %0)
  %4 = call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16.p3(ptr addrspace(3) %0)
  %5 = call i32 @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x1.trans.b16.p3(ptr addrspace(3) %0)
  %6 = call { i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x2.trans.b16.p3(ptr addrspace(3) %0)
  %7 = call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.trans.b16.p3(ptr addrspace(3) %0)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y() #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.tid.z() #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.z() #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.nctaid.x() #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.warpsize() #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.laneid() #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.clusterid.x() #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.clusterid.y() #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.clusterid.z() #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.nclusterid.x() #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.nclusterid.y() #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.nclusterid.z() #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.cluster.ctaid.x() #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.cluster.ctaid.y() #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.cluster.ctaid.z() #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.cluster.nctaid.x() #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.cluster.nctaid.y() #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.cluster.nctaid.z() #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.cluster.ctarank() #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.cluster.nctarank() #0

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare float @llvm.nvvm.rcp.approx.ftz.f(float) #1

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier0() #2

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare i32 @llvm.nvvm.shfl.sync.bfly.i32(i32, i32, i32, i32) #3

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare float @llvm.nvvm.shfl.sync.bfly.f32(i32, float, i32, i32) #3

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare i32 @llvm.nvvm.shfl.sync.up.i32(i32, i32, i32, i32) #3

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare float @llvm.nvvm.shfl.sync.up.f32(i32, float, i32, i32) #3

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare i32 @llvm.nvvm.shfl.sync.down.i32(i32, i32, i32, i32) #3

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare float @llvm.nvvm.shfl.sync.down.f32(i32, float, i32, i32) #3

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare i32 @llvm.nvvm.shfl.sync.idx.i32(i32, i32, i32, i32) #3

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare float @llvm.nvvm.shfl.sync.idx.f32(i32, float, i32, i32) #3

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare { i32, i1 } @llvm.nvvm.shfl.sync.bfly.i32p(i32, i32, i32, i32) #3

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare { float, i1 } @llvm.nvvm.shfl.sync.bfly.f32p(i32, float, i32, i32) #3

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare { i32, i1 } @llvm.nvvm.shfl.sync.up.i32p(i32, i32, i32, i32) #3

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare { float, i1 } @llvm.nvvm.shfl.sync.up.f32p(i32, float, i32, i32) #3

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare { i32, i1 } @llvm.nvvm.shfl.sync.down.i32p(i32, i32, i32, i32) #3

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare { float, i1 } @llvm.nvvm.shfl.sync.down.f32p(i32, float, i32, i32) #3

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare { i32, i1 } @llvm.nvvm.shfl.sync.idx.i32p(i32, i32, i32, i32) #3

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare { float, i1 } @llvm.nvvm.shfl.sync.idx.f32p(i32, float, i32, i32) #3

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare i32 @llvm.nvvm.vote.ballot.sync(i32, i1) #3

; Function Attrs: nocallback nounwind memory(none)
declare { float, float, float, float, float, float, float, float } @llvm.nvvm.mma.m8n8k4.row.col.f32.f32(<2 x half>, <2 x half>, <2 x half>, <2 x half>, float, float, float, float, float, float, float, float) #4

; Function Attrs: nocallback nounwind memory(none)
declare { <2 x half>, <2 x half> } @llvm.nvvm.mma.m16n8k16.row.col.f16.f16(<2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>) #4

; Function Attrs: nocallback nounwind memory(none)
declare { float, float, float, float } @llvm.nvvm.mma.m16n8k16.row.col.f32.f16(<2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>) #4

; Function Attrs: nocallback nounwind memory(none)
declare { <2 x half>, <2 x half> } @llvm.nvvm.mma.m16n8k16.row.col.f16.f32(<2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, float, float, float, float) #4

; Function Attrs: nocallback nounwind memory(none)
declare { float, float, float, float } @llvm.nvvm.mma.m16n8k16.row.col.f32.f32(<2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, float, float, float, float) #4

; Function Attrs: nocallback nounwind memory(none)
declare { i32, i32, i32, i32 } @llvm.nvvm.mma.m16n8k16.row.col.s8(i32, i32, i32, i32, i32, i32, i32) #4

; Function Attrs: nocallback nounwind memory(none)
declare { i32, i32, i32, i32 } @llvm.nvvm.mma.m16n8k16.row.col.satfinite.s8.u8(i32, i32, i32, i32, i32, i32, i32) #4

; Function Attrs: nocallback nounwind memory(none)
declare { i32, i32, i32, i32 } @llvm.nvvm.mma.xor.popc.m16n8k128.row.col.b1(i32, i32, i32, i32, i32, i32, i32) #4

; Function Attrs: nocallback nounwind memory(none)
declare { i32, i32, i32, i32 } @llvm.nvvm.mma.m16n8k32.row.col.satfinite.s4(i32, i32, i32, i32, i32, i32, i32) #4

; Function Attrs: nocallback nounwind memory(none)
declare { double, double } @llvm.nvvm.mma.m8n8k4.row.col.f64(double, double, double, double) #4

; Function Attrs: nocallback nounwind memory(none)
declare { float, float, float, float } @llvm.nvvm.mma.m16n8k4.row.col.tf32(i32, i32, i32, float, float, float, float) #4

; Function Attrs: nocallback nounwind memory(argmem: read)
declare { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.load.a.row.stride.f16.p3(ptr addrspace(3) nocapture readonly, i32) #5

; Function Attrs: nocallback nounwind memory(argmem: write)
declare void @llvm.nvvm.wmma.m16n16k16.store.d.row.stride.f16.p3(ptr addrspace(3) nocapture writeonly, <2 x half>, <2 x half>, <2 x half>, <2 x half>, i32) #6

; Function Attrs: nocallback nounwind memory(none)
declare { <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.mma.row.row.f16.f16(<2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>) #4

; Function Attrs: nocallback nounwind memory(argmem: read)
declare { i32, i32, i32, i32 } @llvm.nvvm.wmma.m16n16k8.load.a.row.stride.tf32.p0(ptr nocapture readonly, i32) #5

; Function Attrs: nocallback nounwind memory(none)
declare { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k8.mma.row.row.tf32(i32, i32, i32, i32, i32, i32, i32, i32, float, float, float, float, float, float, float, float) #4

; Function Attrs: nocallback nounwind memory(argmem: readwrite)
declare void @llvm.nvvm.cp.async.ca.shared.global.4(ptr addrspace(3) noalias writeonly, ptr addrspace(1) noalias readonly) #7

; Function Attrs: nocallback nounwind memory(argmem: readwrite)
declare void @llvm.nvvm.cp.async.ca.shared.global.8(ptr addrspace(3) noalias writeonly, ptr addrspace(1) noalias readonly) #7

; Function Attrs: nocallback nounwind memory(argmem: readwrite)
declare void @llvm.nvvm.cp.async.ca.shared.global.16(ptr addrspace(3) noalias writeonly, ptr addrspace(1) noalias readonly) #7

; Function Attrs: nocallback nounwind memory(argmem: readwrite)
declare void @llvm.nvvm.cp.async.cg.shared.global.16(ptr addrspace(3) noalias writeonly, ptr addrspace(1) noalias readonly) #7

; Function Attrs: nounwind
declare void @llvm.nvvm.cp.async.commit.group() #8

; Function Attrs: nounwind
declare void @llvm.nvvm.cp.async.wait.group(i32 immarg) #8

; Function Attrs: nocallback nounwind memory(argmem: read)
declare i32 @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x1.b16.p3(ptr addrspace(3) nocapture readonly) #5

; Function Attrs: nocallback nounwind memory(argmem: read)
declare { i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x2.b16.p3(ptr addrspace(3) nocapture readonly) #5

; Function Attrs: nocallback nounwind memory(argmem: read)
declare { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16.p3(ptr addrspace(3) nocapture readonly) #5

; Function Attrs: nocallback nounwind memory(argmem: read)
declare i32 @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x1.trans.b16.p3(ptr addrspace(3) nocapture readonly) #5

; Function Attrs: nocallback nounwind memory(argmem: read)
declare { i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x2.trans.b16.p3(ptr addrspace(3) nocapture readonly) #5

; Function Attrs: nocallback nounwind memory(argmem: read)
declare { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.trans.b16.p3(ptr addrspace(3) nocapture readonly) #5
