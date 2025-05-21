; Test to make sure NVVM intrinsics are automatically upgraded.
; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

declare i32 @llvm.nvvm.brev32(i32)
declare i64 @llvm.nvvm.brev64(i64)
declare i32 @llvm.nvvm.clz.i(i32)
declare i32 @llvm.nvvm.clz.ll(i64)
declare i32 @llvm.nvvm.popc.i(i32)
declare i32 @llvm.nvvm.popc.ll(i64)
declare float @llvm.nvvm.h2f(i16)

declare i32 @llvm.nvvm.abs.i(i32)
declare i64 @llvm.nvvm.abs.ll(i64)

declare float @llvm.nvvm.fabs.f(float)
declare float @llvm.nvvm.fabs.ftz.f(float)
declare double @llvm.nvvm.fabs.d(double)

declare i16 @llvm.nvvm.max.s(i16, i16)
declare i32 @llvm.nvvm.max.i(i32, i32)
declare i64 @llvm.nvvm.max.ll(i64, i64)
declare i16 @llvm.nvvm.max.us(i16, i16)
declare i32 @llvm.nvvm.max.ui(i32, i32)
declare i64 @llvm.nvvm.max.ull(i64, i64)
declare i16 @llvm.nvvm.min.s(i16, i16)
declare i32 @llvm.nvvm.min.i(i32, i32)
declare i64 @llvm.nvvm.min.ll(i64, i64)
declare i16 @llvm.nvvm.min.us(i16, i16)
declare i32 @llvm.nvvm.min.ui(i32, i32)
declare i64 @llvm.nvvm.min.ull(i64, i64)

declare i32 @llvm.nvvm.bitcast.f2i(float)
declare float @llvm.nvvm.bitcast.i2f(i32)
declare i64 @llvm.nvvm.bitcast.d2ll(double)
declare double @llvm.nvvm.bitcast.ll2d(i64)

declare i32 @llvm.nvvm.rotate.b32(i32, i32)
declare i64 @llvm.nvvm.rotate.right.b64(i64, i32)
declare i64 @llvm.nvvm.rotate.b64(i64, i32)
declare i64 @llvm.nvvm.swap.lo.hi.b64(i64)

declare ptr addrspace(1) @llvm.nvvm.ptr.gen.to.global.p1.p0(ptr)
declare ptr addrspace(3) @llvm.nvvm.ptr.gen.to.shared.p3.p0(ptr)
declare ptr addrspace(4) @llvm.nvvm.ptr.gen.to.constant.p4.p0(ptr)
declare ptr addrspace(5) @llvm.nvvm.ptr.gen.to.local.p5.p0(ptr)
declare ptr addrspace(101) @llvm.nvvm.ptr.gen.to.param.p101.p0(ptr)
declare ptr @llvm.nvvm.ptr.global.to.gen.p0.p1(ptr addrspace(1))
declare ptr @llvm.nvvm.ptr.shared.to.gen.p0.p3(ptr addrspace(3))
declare ptr @llvm.nvvm.ptr.constant.to.gen.p0.p4(ptr addrspace(4))
declare ptr @llvm.nvvm.ptr.local.to.gen.p0.p5(ptr addrspace(5))
declare ptr @llvm.nvvm.ptr.param.to.gen.p0.p101(ptr addrspace(101))

declare i32 @llvm.nvvm.ldg.global.i.i32.p1(ptr addrspace(1), i32)
declare ptr @llvm.nvvm.ldg.global.p.p1(ptr addrspace(1), i32)
declare float @llvm.nvvm.ldg.global.f.f32.p1(ptr addrspace(1), i32)
declare i32 @llvm.nvvm.ldg.global.i.i32.p0(ptr, i32)
declare ptr @llvm.nvvm.ldg.global.p.p0(ptr, i32)
declare float @llvm.nvvm.ldg.global.f.f32.p0(ptr, i32)

declare i32 @llvm.nvvm.atomic.load.inc.32.p0(ptr, i32)
declare i32 @llvm.nvvm.atomic.load.dec.32.p0(ptr, i32)
declare i32 @llvm.nvvm.atomic.load.add.f32.p0(ptr, float)
declare i32 @llvm.nvvm.atomic.load.add.f64.p0(ptr, double)

declare ptr addrspace(3) @llvm.nvvm.mapa.shared.cluster(ptr addrspace(3), i32)

declare void @llvm.nvvm.cp.async.bulk.global.to.shared.cluster(ptr addrspace(3), ptr addrspace(3), ptr addrspace(1), i32, i16, i64, i1, i1)
declare void @llvm.nvvm.cp.async.bulk.shared.cta.to.cluster(ptr addrspace(3), ptr addrspace(3), ptr addrspace(3), i32)

declare void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.1d(ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr %tm, i32 %d0, i16 %mc, i64 %ch, i1 %f1, i1 %f2);
declare void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr %tm, i32 %d0, i32 %d1, i16 %mc, i64 %ch, i1 %f1, i1 %f2);
declare void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.3d(ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr %tm, i32 %d0, i32 %d1, i32 %d2, i16 %mc, i64 %ch, i1 %f1, i1 %f2);
declare void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.4d(ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr %tm, i32 %d0, i32 %d1, i32 %d2, i32 %d3, i16 %mc, i64 %ch, i1 %f1, i1 %f2);
declare void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.5d(ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr %tm, i32 %d0, i32 %d1, i32 %d2, i32 %d3, i32 %d4, i16 %mc, i64 %ch, i1 %f1, i1 %f2);

declare void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.3d(ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr %tm, i32 %d0, i32 %d1, i32 %d2, i16 %im2col0, i16 %mc, i64 %ch, i1 %f1, i1 %f2);
declare void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.4d(ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr %tm, i32 %d0, i32 %d1, i32 %d2, i32 %d3, i16 %im2col0, i16 %im2col1, i16 %mc, i64 %ch, i1 %f1, i1 %f2);
declare void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.5d(ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr %tm, i32 %d0, i32 %d1, i32 %d2, i32 %d3, i32 %d4, i16 %im2col0, i16 %im2col1, i16 %im2col2, i16 %mc, i64 %ch, i1 %f1, i1 %f2);

declare void @llvm.nvvm.barrier0()
declare void @llvm.nvvm.barrier.n(i32)
declare void @llvm.nvvm.bar.sync(i32)
declare void @llvm.nvvm.barrier(i32, i32)
declare void @llvm.nvvm.barrier.sync(i32)
declare void @llvm.nvvm.barrier.sync.cnt(i32, i32)

; CHECK-LABEL: @simple_upgrade
define void @simple_upgrade(i32 %a, i64 %b, i16 %c) {
; CHECK: call i32 @llvm.bitreverse.i32(i32 %a)
  %r1 = call i32 @llvm.nvvm.brev32(i32 %a)

; CHECK: call i64 @llvm.bitreverse.i64(i64 %b)
  %r2 = call i64 @llvm.nvvm.brev64(i64 %b)

; CHECK: call i32 @llvm.ctlz.i32(i32 %a, i1 false)
  %r3 = call i32 @llvm.nvvm.clz.i(i32 %a)

; CHECK: [[clz:%[a-zA-Z0-9.]+]] = call i64 @llvm.ctlz.i64(i64 %b, i1 false)
; CHECK: trunc i64 [[clz]] to i32
  %r4 = call i32 @llvm.nvvm.clz.ll(i64 %b)

; CHECK: call i32 @llvm.ctpop.i32(i32 %a)
  %r5 = call i32 @llvm.nvvm.popc.i(i32 %a)

; CHECK: [[popc:%[a-zA-Z0-9.]+]] = call i64 @llvm.ctpop.i64(i64 %b)
; CHECK: trunc i64 [[popc]] to i32
  %r6 = call i32 @llvm.nvvm.popc.ll(i64 %b)

; CHECK: call float @llvm.convert.from.fp16.f32(i16 %c)
  %r7 = call float @llvm.nvvm.h2f(i16 %c)
  ret void
}

; CHECK-LABEL: @abs
define void @abs(i32 %a, i64 %b) {
; CHECK-DAG: [[negi:%[a-zA-Z0-9.]+]] = sub i32 0, %a
; CHECK-DAG: [[cmpi:%[a-zA-Z0-9.]+]] = icmp sge i32 %a, 0
; CHECK: select i1 [[cmpi]], i32 %a, i32 [[negi]]
  %r1 = call i32 @llvm.nvvm.abs.i(i32 %a)

; CHECK-DAG: [[negll:%[a-zA-Z0-9.]+]] = sub i64 0, %b
; CHECK-DAG: [[cmpll:%[a-zA-Z0-9.]+]] = icmp sge i64 %b, 0
; CHECK: select i1 [[cmpll]], i64 %b, i64 [[negll]]
  %r2 = call i64 @llvm.nvvm.abs.ll(i64 %b)

  ret void
}

; CHECK-LABEL: @fabs
define void @fabs(float %a, double %b) {
; CHECK: call float @llvm.nvvm.fabs.f32(float %a)
; CHECK: call float @llvm.nvvm.fabs.ftz.f32(float %a)
; CHECK: call double @llvm.nvvm.fabs.f64(double %b)
  %r1 = call float @llvm.nvvm.fabs.f(float %a)
  %r2 = call float @llvm.nvvm.fabs.ftz.f(float %a)
  %r3 = call double @llvm.nvvm.fabs.d(double %b)
  ret void
}

; CHECK-LABEL: @min_max
define void @min_max(i16 %a1, i16 %a2, i32 %b1, i32 %b2, i64 %c1, i64 %c2) {
; CHECK: [[maxs:%[a-zA-Z0-9.]+]] = icmp sge i16 %a1, %a2
; CHECK: select i1 [[maxs]], i16 %a1, i16 %a2
  %r1 = call i16 @llvm.nvvm.max.s(i16 %a1, i16 %a2)

; CHECK: [[maxi:%[a-zA-Z0-9.]+]] = icmp sge i32 %b1, %b2
; CHECK: select i1 [[maxi]], i32 %b1, i32 %b2
  %r2 = call i32 @llvm.nvvm.max.i(i32 %b1, i32 %b2)

; CHECK: [[maxll:%[a-zA-Z0-9.]+]] = icmp sge i64 %c1, %c2
; CHECK: select i1 [[maxll]], i64 %c1, i64 %c2
  %r3 = call i64 @llvm.nvvm.max.ll(i64 %c1, i64 %c2)

; CHECK: [[maxus:%[a-zA-Z0-9.]+]] = icmp uge i16 %a1, %a2
; CHECK: select i1 [[maxus]], i16 %a1, i16 %a2
  %r4 = call i16 @llvm.nvvm.max.us(i16 %a1, i16 %a2)

; CHECK: [[maxui:%[a-zA-Z0-9.]+]] = icmp uge i32 %b1, %b2
; CHECK: select i1 [[maxui]], i32 %b1, i32 %b2
  %r5 = call i32 @llvm.nvvm.max.ui(i32 %b1, i32 %b2)

; CHECK: [[maxull:%[a-zA-Z0-9.]+]] = icmp uge i64 %c1, %c2
; CHECK: select i1 [[maxull]], i64 %c1, i64 %c2
  %r6 = call i64 @llvm.nvvm.max.ull(i64 %c1, i64 %c2)

; CHECK: [[mins:%[a-zA-Z0-9.]+]] = icmp sle i16 %a1, %a2
; CHECK: select i1 [[mins]], i16 %a1, i16 %a2
  %r7 = call i16 @llvm.nvvm.min.s(i16 %a1, i16 %a2)

; CHECK: [[mini:%[a-zA-Z0-9.]+]] = icmp sle i32 %b1, %b2
; CHECK: select i1 [[mini]], i32 %b1, i32 %b2
  %r8 = call i32 @llvm.nvvm.min.i(i32 %b1, i32 %b2)

; CHECK: [[minll:%[a-zA-Z0-9.]+]] = icmp sle i64 %c1, %c2
; CHECK: select i1 [[minll]], i64 %c1, i64 %c2
  %r9 = call i64 @llvm.nvvm.min.ll(i64 %c1, i64 %c2)

; CHECK: [[minus:%[a-zA-Z0-9.]+]] = icmp ule i16 %a1, %a2
; CHECK: select i1 [[minus]], i16 %a1, i16 %a2
  %r10 = call i16 @llvm.nvvm.min.us(i16 %a1, i16 %a2)

; CHECK: [[minui:%[a-zA-Z0-9.]+]] = icmp ule i32 %b1, %b2
; CHECK: select i1 [[minui]], i32 %b1, i32 %b2
  %r11 = call i32 @llvm.nvvm.min.ui(i32 %b1, i32 %b2)

; CHECK: [[minull:%[a-zA-Z0-9.]+]] = icmp ule i64 %c1, %c2
; CHECK: select i1 [[minull]], i64 %c1, i64 %c2
  %r12 = call i64 @llvm.nvvm.min.ull(i64 %c1, i64 %c2)

  ret void
}

; CHECK-LABEL: @bitcast
define void @bitcast(i32 %a, i64 %b, float %c, double %d) {
; CHECK: bitcast float %c to i32
; CHECK: bitcast i32 %a to float
; CHECK: bitcast double %d to i64
; CHECK: bitcast i64 %b to double
;
  %r1 = call i32 @llvm.nvvm.bitcast.f2i(float %c)
  %r2 = call float @llvm.nvvm.bitcast.i2f(i32 %a)
  %r3 = call i64 @llvm.nvvm.bitcast.d2ll(double %d)
  %r4 = call double @llvm.nvvm.bitcast.ll2d(i64 %b)

  ret void
}

; CHECK-LABEL: @rotate
define void @rotate(i32 %a, i64 %b) {
; CHECK: call i32 @llvm.fshl.i32(i32 %a, i32 %a, i32 6)
; CHECK: call i64 @llvm.fshr.i64(i64 %b, i64 %b, i64 7)
; CHECK: call i64 @llvm.fshl.i64(i64 %b, i64 %b, i64 8)
; CHECK: call i64 @llvm.fshl.i64(i64 %b, i64 %b, i64 32)
;
  %r1 = call i32 @llvm.nvvm.rotate.b32(i32 %a, i32 6)
  %r2 = call i64 @llvm.nvvm.rotate.right.b64(i64 %b, i32 7)
  %r3 = call i64 @llvm.nvvm.rotate.b64(i64 %b, i32 8)
  %r4 = call i64 @llvm.nvvm.swap.lo.hi.b64(i64 %b)
  ret void
}

; CHECK-LABEL: @addrspacecast
define void @addrspacecast(ptr %p0) {
; CHECK: %1 = addrspacecast ptr %p0 to ptr addrspace(1)
; CHECK: %2 = addrspacecast ptr addrspace(1) %1 to ptr
; CHECK: %3 = addrspacecast ptr %2 to ptr addrspace(3)
; CHECK: %4 = addrspacecast ptr addrspace(3) %3 to ptr
; CHECK: %5 = addrspacecast ptr %4 to ptr addrspace(4)
; CHECK: %6 = addrspacecast ptr addrspace(4) %5 to ptr
; CHECK: %7 = addrspacecast ptr %6 to ptr addrspace(5)
; CHECK: %8 = addrspacecast ptr addrspace(5) %7 to ptr
; CHECK: %9 = addrspacecast ptr %8 to ptr addrspace(101)
; CHECK: %10 = addrspacecast ptr addrspace(101) %9 to ptr
;
  %p1 = call ptr addrspace(1) @llvm.nvvm.ptr.gen.to.global.p1.p0(ptr %p0)
  %p2 = call ptr @llvm.nvvm.ptr.global.to.gen.p0.p1(ptr addrspace(1) %p1)

  %p3 = call ptr addrspace(3) @llvm.nvvm.ptr.gen.to.shared.p3.p0(ptr %p2)
  %p4 = call ptr @llvm.nvvm.ptr.shared.to.gen.p0.p3(ptr addrspace(3) %p3)

  %p5 = call ptr addrspace(4) @llvm.nvvm.ptr.gen.to.constant.p4.p0(ptr %p4)
  %p6 = call ptr @llvm.nvvm.ptr.constant.to.gen.p0.p4(ptr addrspace(4) %p5)

  %p7 = call ptr addrspace(5) @llvm.nvvm.ptr.gen.to.local.p5.p0(ptr %p6)
  %p8 = call ptr @llvm.nvvm.ptr.local.to.gen.p0.p5(ptr addrspace(5) %p7)

  %p9 = call ptr addrspace(101) @llvm.nvvm.ptr.gen.to.param.p101.p0(ptr %p8)
  %p10 = call ptr @llvm.nvvm.ptr.param.to.gen.p0.p101(ptr addrspace(101) %p9)

  ret void
}

; CHECK-LABEL: @ldg
define void @ldg(ptr %p0, ptr addrspace(1) %p1) {
; CHECK: %1 = load i32, ptr addrspace(1) %p1, align 4, !invariant.load !0
; CHECK: %2 = load ptr, ptr addrspace(1) %p1, align 8, !invariant.load !0
; CHECK: %3 = load float, ptr addrspace(1) %p1, align 16, !invariant.load !0

; CHECK: %4 = addrspacecast ptr %p0 to ptr addrspace(1)
; CHECK: %5 = load i32, ptr addrspace(1) %4, align 4, !invariant.load !0
; CHECK: %6 = addrspacecast ptr %p0 to ptr addrspace(1)
; CHECK: %7 = load ptr, ptr addrspace(1) %6, align 8, !invariant.load !0
; CHECK: %8 = addrspacecast ptr %p0 to ptr addrspace(1)
; CHECK: %9 = load float, ptr addrspace(1) %8, align 16, !invariant.load !0
;
  %v1 = call i32 @llvm.nvvm.ldg.global.i.i32.p1(ptr addrspace(1) %p1, i32 4)
  %v2 = call ptr @llvm.nvvm.ldg.global.p.p1(ptr addrspace(1) %p1, i32 8 )
  %v3 = call float @llvm.nvvm.ldg.global.f.f32.p1(ptr addrspace(1) %p1, i32 16)

  %v4 = call i32 @llvm.nvvm.ldg.global.i.i32.p0(ptr %p0, i32 4)
  %v5 = call ptr @llvm.nvvm.ldg.global.p.p0(ptr %p0, i32 8)
  %v6 = call float @llvm.nvvm.ldg.global.f.f32.p0(ptr %p0, i32 16)

  ret void
}

; CHECK-LABEL: @atomics
define i32 @atomics(ptr %p0, i32 %a, float %b, double %c) {
; CHECK: %1 = atomicrmw uinc_wrap ptr %p0, i32 %a seq_cst
; CHECK: %2 = atomicrmw udec_wrap ptr %p0, i32 %a seq_cst
; CHECK: %3 = atomicrmw fadd ptr %p0, float %b seq_cst
; CHECK: %4 = atomicrmw fadd ptr %p0, double %c seq_cst

  %r1 = call i32 @llvm.nvvm.atomic.load.inc.32.p0(ptr %p0, i32 %a)
  %r2 = call i32 @llvm.nvvm.atomic.load.dec.32.p0(ptr %p0, i32 %a)
  %r3 = call float @llvm.nvvm.atomic.load.add.f32.p0(ptr %p0, float %b)
  %r4 = call double @llvm.nvvm.atomic.load.add.f64.p0(ptr %p0, double %c)
  ret i32 %r2
}

; CHECK-LABEL: @nvvm_shared_cluster_intrinsics
define void @nvvm_shared_cluster_intrinsics(ptr addrspace(3) %p0, i32 %offset) {
; CHECK: %r = call ptr addrspace(7) @llvm.nvvm.mapa.shared.cluster(ptr addrspace(3) %p0, i32 %offset)
  %r = call ptr addrspace(3) @llvm.nvvm.mapa.shared.cluster(ptr addrspace(3) %p0, i32 %offset)
  ret void
}

; CHECK-LABEL: @nvvm_cp_async_bulk_intrinsics
define void @nvvm_cp_async_bulk_intrinsics(ptr addrspace(3) %dst, ptr addrspace(3) %bar, ptr addrspace(1) %src, ptr addrspace(3) %src_shared, i32 %size) {
; CHECK: call void @llvm.nvvm.cp.async.bulk.global.to.shared.cluster(ptr addrspace(7) %1, ptr addrspace(3) %bar, ptr addrspace(1) %src, i32 %size, i16 0, i64 0, i1 false, i1 false)
; CHECK: call void @llvm.nvvm.cp.async.bulk.shared.cta.to.cluster(ptr addrspace(7) %2, ptr addrspace(3) %bar, ptr addrspace(3) %src_shared, i32 %size)
  call void @llvm.nvvm.cp.async.bulk.global.to.shared.cluster(ptr addrspace(3) %dst, ptr addrspace(3) %bar, ptr addrspace(1) %src, i32 %size, i16 0, i64 0, i1 false, i1 false)
  call void @llvm.nvvm.cp.async.bulk.shared.cta.to.cluster(ptr addrspace(3) %dst, ptr addrspace(3) %bar, ptr addrspace(3) %src_shared, i32 %size)
  ret void
}

; CHECK-LABEL: @nvvm_cp_async_bulk_tensor_g2s_im2col
define void @nvvm_cp_async_bulk_tensor_g2s_im2col(ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i32 %d3, i32 %d4, i16 %im2col0, i16 %im2col1, i16 %im2col2, i16 %mc, i64 %ch) {
; CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.3d(ptr addrspace(7) %1, ptr addrspace(3) %bar, ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i16 %im2col0, i16 0, i64 0, i1 false, i1 false)
; CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.4d(ptr addrspace(7) %2, ptr addrspace(3) %bar, ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i32 %d3, i16 %im2col0, i16 %im2col1, i16 0, i64 0, i1 false, i1 false)
; CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.5d(ptr addrspace(7) %3, ptr addrspace(3) %bar, ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i32 %d3, i32 %d4, i16 %im2col0, i16 %im2col1, i16 %im2col2, i16 0, i64 0, i1 false, i1 false)
  call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.3d(ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i16 %im2col0, i16 0, i64 0, i1 0, i1 0)
  call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.4d(ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i32 %d3, i16 %im2col0, i16 %im2col1, i16 0, i64 0, i1 0, i1 0)
  call void @llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.5d(ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i32 %d3, i32 %d4, i16 %im2col0, i16 %im2col1, i16 %im2col2, i16 0, i64 0, i1 0, i1 0)
  ret void
}

; CHECK-LABEL: @nvvm_cp_async_bulk_tensor_g2s_tile
define void @nvvm_cp_async_bulk_tensor_g2s_tile(ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i32 %d3, i32 %d4, i16 %mc, i64 %ch) {
; CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.1d(ptr addrspace(7) %1, ptr addrspace(3) %bar, ptr %tmap, i32 %d0, i16 0, i64 0, i1 false, i1 false)
; CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(ptr addrspace(7) %2, ptr addrspace(3) %bar, ptr %tmap, i32 %d0, i32 %d1, i16 0, i64 0, i1 false, i1 false)
; CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.3d(ptr addrspace(7) %3, ptr addrspace(3) %bar, ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i16 0, i64 0, i1 false, i1 false)
; CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.4d(ptr addrspace(7) %4, ptr addrspace(3) %bar, ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i32 %d3, i16 0, i64 0, i1 false, i1 false)
; CHECK: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.5d(ptr addrspace(7) %5, ptr addrspace(3) %bar, ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i32 %d3, i32 %d4, i16 0, i64 0, i1 false, i1 false)
  call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.1d(ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr %tmap, i32 %d0, i16 0, i64 0, i1 0, i1 0)
  call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr %tmap, i32 %d0, i32 %d1, i16 0, i64 0, i1 0, i1 0)
  call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.3d(ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i16 0, i64 0, i1 0, i1 0)
  call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.4d(ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i32 %d3, i16 0, i64 0, i1 0, i1 0)
  call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.5d(ptr addrspace(3) %d, ptr addrspace(3) %bar, ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i32 %d3, i32 %d4, i16 0, i64 0, i1 0, i1 0)
  ret void
}

define void @cta_barriers(i32 %x, i32 %y) {
; CHECK: call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
; CHECK: call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 %x)
; CHECK: call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 %x)
; CHECK: call void @llvm.nvvm.barrier.cta.sync.aligned(i32 %x, i32 %y)
; CHECK: call void @llvm.nvvm.barrier.cta.sync.all(i32 %x)
; CHECK: call void @llvm.nvvm.barrier.cta.sync(i32 %x, i32 %y)
  call void @llvm.nvvm.barrier0()
  call void @llvm.nvvm.barrier.n(i32 %x)
  call void @llvm.nvvm.bar.sync(i32 %x)
  call void @llvm.nvvm.barrier(i32 %x, i32 %y)
  call void @llvm.nvvm.barrier.sync(i32 %x)
  call void @llvm.nvvm.barrier.sync.cnt(i32 %x, i32 %y)
  ret void
}
