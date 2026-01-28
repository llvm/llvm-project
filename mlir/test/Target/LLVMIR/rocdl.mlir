// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

llvm.func @rocdl_special_regs() -> i32 {
  // CHECK-LABEL: rocdl_special_regs
  // CHECK: call i32 @llvm.amdgcn.workitem.id.x()
  %1 = rocdl.workitem.id.x : i32
  // CHECK: call i32 @llvm.amdgcn.workitem.id.y()
  %2 = rocdl.workitem.id.y : i32
  // CHECK: call i32 @llvm.amdgcn.workitem.id.z()
  %3 = rocdl.workitem.id.z : i32
  // CHECK: call i32 @llvm.amdgcn.workgroup.id.x()
  %4 = rocdl.workgroup.id.x : i32
  // CHECK: call i32 @llvm.amdgcn.workgroup.id.y()
  %5 = rocdl.workgroup.id.y : i32
  // CHECK: call i32 @llvm.amdgcn.workgroup.id.z()
  %6 = rocdl.workgroup.id.z : i32
  // CHECK: call i32 @llvm.amdgcn.cluster.id.x()
  %7 = rocdl.cluster.id.x : i32
  // CHECK: call i32 @llvm.amdgcn.cluster.id.y()
  %8 = rocdl.cluster.id.y : i32
  // CHECK: call i32 @llvm.amdgcn.cluster.id.z()
  %9 = rocdl.cluster.id.z : i32
  // CHECK: call i64 @__ockl_get_local_size(i32 0)
  %10 = rocdl.workgroup.dim.x : i64
  // CHECK: call i64 @__ockl_get_local_size(i32 1)
  %11 = rocdl.workgroup.dim.y : i64
  // CHECK: call i64 @__ockl_get_local_size(i32 2)
  %12 = rocdl.workgroup.dim.z : i64
  // CHECK: call i64 @__ockl_get_num_groups(i32 0)
  %13 = rocdl.grid.dim.x : i64
  // CHECK: call i64 @__ockl_get_num_groups(i32 1)
  %14 = rocdl.grid.dim.y : i64
  // CHECK: call i64 @__ockl_get_num_groups(i32 2)
  %15 = rocdl.grid.dim.z : i64

  // CHECK: call range(i32 0, 64) i32 @llvm.amdgcn.workitem.id.x()
  %16 = rocdl.workitem.id.x range <i32, 0, 64> : i32

  // CHECK: call range(i64 1, 65) i64 @__ockl_get_local_size(i32 0)
  %17 = rocdl.workgroup.dim.x range <i32, 1, 65> : i64

  // CHECK: call i32 @llvm.amdgcn.wave.id()
  %18 = rocdl.wave.id : i32

  // CHECK: call range(i32 32, 65) i32 @llvm.amdgcn.wave.id()
  %19 = rocdl.wave.id range <i32, 32, 65> : i32

  // CHECK: call i32 @llvm.amdgcn.wavefrontsize()
  %20 = rocdl.wavefrontsize : i32

  // CHECK: call range(i32 32, 65) i32 @llvm.amdgcn.wavefrontsize()
  %21 = rocdl.wavefrontsize range <i32, 32, 65> : i32

  llvm.return %1 : i32
}

llvm.func @kernel_func() attributes {rocdl.kernel} {
  // CHECK-LABEL: amdgpu_kernel void @kernel_func()
  // CHECK: #[[$KERNEL_ATTRS:[0-9]+]]
  llvm.return
}

llvm.func @kernel_func_workgroups()
    attributes {rocdl.kernel, rocdl.max_flat_work_group_size = 1024 : index} {
  // CHECK-LABEL: amdgpu_kernel void @kernel_func_workgroups()
  // CHECK: #[[$KERNEL_WORKGROUP_ATTRS:[0-9]+]]
  llvm.return
}

llvm.func @kernel_math_ops(%a: f32, %b: f16, %c: bf16) {
  // CHECK-LABEL: kernel_math_ops
  // CHECK: call float @llvm.amdgcn.tanh.f32(float %{{.*}})
  // CHECK: call half @llvm.amdgcn.tanh.f16(half %{{.*}})
  // CHECK: call bfloat @llvm.amdgcn.tanh.bf16(bfloat %{{.*}})
  %tanh0 = rocdl.tanh %a f32 -> f32
  %tanh1 = rocdl.tanh %b f16 -> f16
  %tanh2 = rocdl.tanh %c bf16 -> bf16

  // CHECK: call float @llvm.amdgcn.sin.f32(float %{{.*}})
  // CHECK: call half @llvm.amdgcn.sin.f16(half %{{.*}})
  // CHECK: call bfloat @llvm.amdgcn.sin.bf16(bfloat %{{.*}})
  %sin0 = rocdl.sin %a f32 -> f32
  %sin1 = rocdl.sin %b f16 -> f16
  %sin2 = rocdl.sin %c bf16 -> bf16

  // CHECK: call float @llvm.amdgcn.cos.f32(float %{{.*}})
  // CHECK: call half @llvm.amdgcn.cos.f16(half %{{.*}})
  // CHECK: call bfloat @llvm.amdgcn.cos.bf16(bfloat %{{.*}})
  %cos0 = rocdl.cos %a f32 -> f32
  %cos1 = rocdl.cos %b f16 -> f16
  %cos2 = rocdl.cos %c bf16 -> bf16

  // CHECK: call float @llvm.amdgcn.rcp.f32(float %{{.*}})
  // CHECK: call half @llvm.amdgcn.rcp.f16(half %{{.*}})
  // CHECK: call bfloat @llvm.amdgcn.rcp.bf16(bfloat %{{.*}})
  %rcp0 = rocdl.rcp %a f32 -> f32
  %rcp1 = rocdl.rcp %b f16 -> f16
  %rcp2 = rocdl.rcp %c bf16 -> bf16

  // CHECK: call float @llvm.amdgcn.exp2.f32(float %{{.*}})
  // CHECK: call half @llvm.amdgcn.exp2.f16(half %{{.*}})
  // CHECK: call bfloat @llvm.amdgcn.exp2.bf16(bfloat %{{.*}})
  %exp2_0 = rocdl.exp2 %a f32 -> f32
  %exp2_1 = rocdl.exp2 %b f16 -> f16
  %exp2_2 = rocdl.exp2 %c bf16 -> bf16

  // CHECK: call float @llvm.amdgcn.log.f32(float %{{.*}})
  // CHECK: call half @llvm.amdgcn.log.f16(half %{{.*}})
  // CHECK: call bfloat @llvm.amdgcn.log.bf16(bfloat %{{.*}})
  %log0 = rocdl.log %a f32 -> f32
  %log1 = rocdl.log %b f16 -> f16
  %log2 = rocdl.log %c bf16 -> bf16

  // CHECK: call float @llvm.amdgcn.sqrt.f32(float %{{.*}})
  // CHECK: call half @llvm.amdgcn.sqrt.f16(half %{{.*}})
  // CHECK: call bfloat @llvm.amdgcn.sqrt.bf16(bfloat %{{.*}})
  %sqrt0 = rocdl.sqrt %a f32 -> f32
  %sqrt1 = rocdl.sqrt %b f16 -> f16
  %sqrt2 = rocdl.sqrt %c bf16 -> bf16

  // CHECK: call float @llvm.amdgcn.rsq.f32(float %{{.*}})
  // CHECK: call half @llvm.amdgcn.rsq.f16(half %{{.*}})
  // CHECK: call bfloat @llvm.amdgcn.rsq.bf16(bfloat %{{.*}})
  %rsq0 = rocdl.rsq %a f32 -> f32
  %rsq1 = rocdl.rsq %b f16 -> f16
  %rsq2 = rocdl.rsq %c bf16 -> bf16
  llvm.return
}

llvm.func @known_block_sizes()
    attributes {rocdl.kernel,
      rocdl.flat_work_group_size = "128,128",
      rocdl.reqd_work_group_size = array<i32: 16, 4, 2>} {
  // CHECK-LABEL: amdgpu_kernel void @known_block_sizes()
  // CHECK: #[[$KNOWN_BLOCK_SIZE_ATTRS:[0-9]+]]
  // CHECK: !reqd_work_group_size ![[$REQD_WORK_GROUP_SIZE:[0-9]+]]
  llvm.return
}

llvm.func @kernel_func_no_uniform_work_groups() attributes {rocdl.kernel, rocdl.uniform_work_group_size = false} {
  // CHECK-LABEL: amdgpu_kernel void @kernel_func_no_uniform_work_groups()
  // CHECK: #[[$KERNEL_NO_UNIFORM_WORK_GROUPS_ATTRS:[0-9]+]]
  llvm.return
}

llvm.func @kernel_func_waves_per_eu()
    attributes {rocdl.kernel, rocdl.waves_per_eu = 2 : i32} {
  // CHECK-LABEL: amdgpu_kernel void @kernel_func_waves_per_eu()
  // CHECK: #[[$KERNEL_WAVES_PER_EU_ATTR:[0-9]+]]
  llvm.return
}

llvm.func @kernel_func_unsafe_fp_atomics()
    attributes {rocdl.kernel, rocdl.unsafe_fp_atomics = true} {
  // CHECK-LABEL: amdgpu_kernel void @kernel_func_unsafe_fp_atomics()
  // CHECK: #[[$KERNEL_UNSAFE_FP_ATOMICS_ATTR:[0-9]+]]
  llvm.return
}

llvm.func @rocdl.lane_id() -> i32 {
  // CHECK: [[mbcntlo:%.+]] = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  // CHECK-NEXT: call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 [[mbcntlo]])
  %0 = llvm.mlir.constant(-1 : i32) : i32
  %1 = llvm.mlir.constant(0 : i32) : i32
  %2 = rocdl.mbcnt.lo %0, %1 {res_attrs = [{llvm.noundef, llvm.range = #llvm.constant_range<i32, 0, 32>}]} : (i32, i32) -> i32
  %3 = rocdl.mbcnt.hi %0, %2 {res_attrs = [{llvm.noundef, llvm.range = #llvm.constant_range<i32, 0, 64>}]} : (i32, i32) -> i32
  llvm.return %3 : i32
}

llvm.func @rocdl.swizzle(%src : i32) -> i32 {
  // CHECK-LABEL: rocdl.swizzle
  // CHECK: call i32 @llvm.amdgcn.ds.swizzle
  %offset = llvm.mlir.constant(100 : i32) : i32
  %0 = rocdl.ds_swizzle %src, %offset : (i32, i32) -> i32
  llvm.return %0 : i32
}

llvm.func @rocdl.bpermute(%src : i32) -> i32 {
  // CHECK-LABEL: rocdl.bpermute
  // CHECK: call i32 @llvm.amdgcn.ds.bpermute
  %index = llvm.mlir.constant(10 : i32) : i32
  %0 = rocdl.ds_bpermute %index, %src : (i32, i32) -> i32
  llvm.return %0 : i32
}

llvm.func @rocdl.ballot32(%pred : i1) -> i32 {
  // CHECK-LABEL: rocdl.ballot32
  // CHECK: call i32 @llvm.amdgcn.ballot
  %0 = rocdl.ballot %pred : i32
  llvm.return %0 : i32
}

llvm.func @rocdl.ballot64(%pred : i1) -> i64 {
  // CHECK-LABEL: rocdl.ballot64
  // CHECK: call i64 @llvm.amdgcn.ballot
  %0 = rocdl.ballot %pred : i64
  llvm.return %0 : i64
}

llvm.func @rocdl.readfirstlane(%src0 : f32, %src1: f64, %src2: i32, %src3: vector<2 x f32>) -> f32 {
  // CHECK-LABEL: rocdl.readfirstlane
  // CHECK: call float @llvm.amdgcn.readfirstlane.f32(float %{{.*}})
  %0 = rocdl.readfirstlane %src0 : f32

  // CHECK: call double @llvm.amdgcn.readfirstlane.f64(double %{{.*}})
  %1 = rocdl.readfirstlane %src1 : f64

  // CHECK: call i32 @llvm.amdgcn.readfirstlane.i32(i32 %{{.*}})
  %2 = rocdl.readfirstlane %src2 : i32

  // CHECK: call <2 x float> @llvm.amdgcn.readfirstlane.v2f32(<2 x float> %{{.*}})
  %3 = rocdl.readfirstlane %src3 : vector<2 x f32>

  llvm.return %0 : f32
}

llvm.func @rocdl.readlane(%src0 : f32, %src1: f64, %src2: i32, %src3: vector<2 x f32>) -> f32 {
  %idx = llvm.mlir.constant(0 : i32) : i32

  // CHECK-LABEL: rocdl.readlane
  // CHECK: call float @llvm.amdgcn.readlane.f32(float %{{.*}}, i32 0)
  %0 = rocdl.readlane %src0, %idx : (f32, i32) -> f32

  // CHECK: call double @llvm.amdgcn.readlane.f64(double %{{.*}}, i32 0)
  %1 = rocdl.readlane %src1, %idx : (f64, i32) -> f64

  // CHECK: call i32 @llvm.amdgcn.readlane.i32(i32 %{{.*}}, i32 0)
  %2 = rocdl.readlane %src2, %idx : (i32, i32) -> i32

  // CHECK: call <2 x float> @llvm.amdgcn.readlane.v2f32(<2 x float> %{{.*}}, i32 0)
  %3 = rocdl.readlane %src3, %idx : (vector<2 x f32>, i32) -> vector<2 x f32>

  llvm.return %0 : f32
}

llvm.func @rocdl.s.waitcnt() {
  // CHECK-LABEL: rocdl.s.waitcnt
  // CHECK-NEXT: call void @llvm.amdgcn.s.waitcnt(i32 0)
  rocdl.s.waitcnt 0
  llvm.return
}

llvm.func @rocdl.s.sleep() {
  // CHECK-LABEL: rocdl.s.sleep
  // CHECK-NEXT: call void @llvm.amdgcn.s.sleep(i32 0)
  rocdl.s.sleep 0
  llvm.return
}

llvm.func @rocdl.s.nop() {
  // CHECK-LABEL: rocdl.s.nop
  // CHECK-NEXT: call void @llvm.amdgcn.s.nop(i16 0)
  rocdl.s.nop 0
  llvm.return
}

llvm.func @rocdl.s.barrier() {
  // CHECK-LABEL: rocdl.s.barrier
  // CHECK-NEXT: call void @llvm.amdgcn.s.barrier()
  rocdl.s.barrier
  llvm.return
}


llvm.func @rocdl.barrier() {
  // CHECK-LABEL: rocdl.barrier
  // CHECK:      fence syncscope("workgroup") release
  // CHECK-NEXT: call void @llvm.amdgcn.s.barrier()
  // CHECK-NEXT: fence syncscope("workgroup") acquire
  rocdl.barrier
  llvm.return
}

llvm.func @rocdl.s.barrier.init(%ptr : !llvm.ptr<3>) {
  // CHECK-LABEL: rocdl.s.barrier.init
  // CHECK: call void @llvm.amdgcn.s.barrier.init(ptr addrspace(3) %{{.*}}, i32 1)
  rocdl.s.barrier.init %ptr member_cnt = 1 : !llvm.ptr<3>
  llvm.return
}

llvm.func @rocdl.s.barrier.signal() {
  // CHECK-LABEL: rocdl.s.barrier.signal
  // CHECK-NEXT: call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  rocdl.s.barrier.signal id = -1
  llvm.return
}

llvm.func @rocdl.s.barrier.signal.var(%ptr : !llvm.ptr<3>) {
  // CHECK-LABEL: rocdl.s.barrier.signal.var
  // CHECK: call void @llvm.amdgcn.s.barrier.signal.var(ptr addrspace(3) %{{.*}}, i32 1)
  rocdl.s.barrier.signal.var %ptr member_cnt = 1 : !llvm.ptr<3>
  llvm.return
}

llvm.func @rocdl.s.barrier.join(%ptr : !llvm.ptr<3>) {
  // CHECK-LABEL: rocdl.s.barrier.join
  // CHECK: call void @llvm.amdgcn.s.barrier.join(ptr addrspace(3) %{{.*}})
  rocdl.s.barrier.join %ptr : !llvm.ptr<3>
  llvm.return
}

llvm.func @rocdl.s.barrier.leave() {
  // CHECK-LABEL: rocdl.s.barrier.leave
  // CHECK: call void @llvm.amdgcn.s.barrier.leave(i16 1)
  rocdl.s.barrier.leave id = 1
  llvm.return
}

llvm.func @rocdl.s.barrier.wait() {
  // CHECK-LABEL: rocdl.s.barrier.wait
  // CHECK-NEXT: call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  rocdl.s.barrier.wait id = -1
  llvm.return
}

llvm.func @rocdl.s.barrier.signal.isfirst() {
  // CHECK-LABEL: rocdl.s.barrier.signal.isfirst
  // CHECK:  %{{.*}} = call i1 @llvm.amdgcn.s.barrier.signal.isfirst(i32 1)
  %0 = rocdl.s.barrier.signal.isfirst id = 1 -> i1
  llvm.return
}

llvm.func @rocdl.s.get.barrier.state() {
  // CHECK-LABEL: rocdl.s.get.barrier.state
  // CHECK: %{{.*}} = call i32 @llvm.amdgcn.s.get.barrier.state(i32 1)
  %0 = rocdl.s.get.barrier.state id = 1 -> i32
  llvm.return
}

llvm.func @rocdl.s.get.named.barrier.state(%ptr : !llvm.ptr<3>) {
  // CHECK-LABEL: rocdl.s.get.named.barrier.state
  // CHECK: %{{.*}} = call i32 @llvm.amdgcn.s.get.named.barrier.state(ptr addrspace(3) %{{.*}})
  %0 = rocdl.s.get.named.barrier.state %ptr : !llvm.ptr<3> -> i32
  llvm.return
}

llvm.func @rocdl.s.wakeup.barrier(%ptr : !llvm.ptr<3>) {
  // CHECK-LABEL: rocdl.s.wakeup.barrier
  // CHECK: call void @llvm.amdgcn.s.wakeup.barrier(ptr addrspace(3) %{{.*}})
  rocdl.s.wakeup.barrier %ptr : !llvm.ptr<3>
  llvm.return
}

llvm.func @rocdl.s.wait.dscnt() {
  // CHECK-LABEL: rocdl.s.wait.dscnt
  // CHECK-NEXT: call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  rocdl.s.wait.dscnt 0
  llvm.return
}

llvm.func @rocdl.s.wait.loadcnt() {
  // CHECK-LABEL: rocdl.s.wait.loadcnt
  // CHECK-NEXT: call void @llvm.amdgcn.s.wait.loadcnt(i16 0)
  rocdl.s.wait.loadcnt 0
  llvm.return
}

llvm.func @rocdl.s.wait.storecnt() {
  // CHECK-LABEL: rocdl.s.wait.storecnt
  // CHECK-NEXT: call void @llvm.amdgcn.s.wait.storecnt(i16 0)
  rocdl.s.wait.storecnt 0
  llvm.return
}

llvm.func @rocdl.s.wait.expcnt() {
  // CHECK-LABEL: rocdl.s.wait.expcnt
  // CHECK-NEXT: call void @llvm.amdgcn.s.wait.expcnt(i16 0)
  rocdl.s.wait.expcnt 0
  llvm.return
}

llvm.func @rocdl.s.wait.asynccnt() {
  // CHECK-LABEL: rocdl.s.wait.asynccnt
  // CHECK-NEXT: call void @llvm.amdgcn.s.wait.asynccnt(i16 0)
  rocdl.s.wait.asynccnt 0
  llvm.return
}

llvm.func @rocdl.s.wait.tensorcnt() {
  // CHECK-LABEL: rocdl.s.wait.tensorcnt
  // CHECK-NEXT: call void @llvm.amdgcn.s.wait.tensorcnt(i16 0)
  rocdl.s.wait.tensorcnt 0
  llvm.return
}

llvm.func @rocdl.setprio() {
  // CHECK: call void @llvm.amdgcn.s.setprio(i16 0)
  rocdl.s.setprio 0
  // CHECK-NEXT: call void @llvm.amdgcn.s.setprio(i16 1)
  rocdl.s.setprio 1
  llvm.return
}

llvm.func @rocdl.schedbarrier() {
  // CHECK: call void @llvm.amdgcn.sched.barrier(i32 0)
  rocdl.sched.barrier 0
  // CHECK-NEXT: call void @llvm.amdgcn.sched.barrier(i32 1)
  rocdl.sched.barrier 1
  llvm.return
}

llvm.func @rocdl.sched.group.barrier() {
  // CHECK-LABEL: rocdl.sched.group.barrier
  // CHECK-NEXT: call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  rocdl.sched.group.barrier 8, 1, 0
  llvm.return
}

llvm.func @rocdl.iglp.opt() {
  // CHECK-LABEL: rocdl.iglp.opt
  // CHECK-NEXT: call void @llvm.amdgcn.iglp.opt(i32 0)
  rocdl.iglp.opt 0
  // CHECK-NEXT: call void @llvm.amdgcn.iglp.opt(i32 1)
  rocdl.iglp.opt 1
  llvm.return
}

llvm.func @rocdl.xdlops(%arg0 : f32, %arg1 : f32,
                   %arg2 : vector<32 x f32>, %arg3: i32,
                   %arg4 : vector<16 x f32>, %arg5 : vector<4xf32>,
                   %arg6 : vector<4xf16>, %arg7 : vector<32 x i32>,
                   %arg8 : vector<16 x i32>, %arg9 : vector<4xi32>,
                   %arg10 : vector<2xi16>, %arg11 : i64,
                   %arg12 : vector<8xbf16>, %arg13 : vector<4xi32>,
                   %arg14 : vector<8xf16>) -> vector<32 x f32> {

  // CHECK-LABEL: rocdl.xdlops
  // CHECK: call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float %{{.*}}, float %{{.*}}, <32 x float> %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  %r0 = rocdl.mfma.f32.32x32x1f32 %arg0, %arg1, %arg2, 0, 0, 0 :
                            (f32, f32, vector<32 x f32>) -> vector<32 x f32>

  // CHECK: call <16 x float> @llvm.amdgcn.mfma.f32.16x16x1f32(float %{{.*}}, float %{{.*}}, <16 x float> %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  %r1 = rocdl.mfma.f32.16x16x1f32 %arg0, %arg1, %arg4, 0, 0, 0 :
                            (f32, f32, vector<16 x f32>) -> vector<16 x f32>

  // CHECK: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %{{.*}}, float %{{.*}}, <4 x float> %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  %r2 = rocdl.mfma.f32.16x16x4f32 %arg0, %arg1, %arg5, 0, 0, 0 :
                            (f32, f32, vector<4xf32>) -> vector<4xf32>

  // CHECK: call <4 x float> @llvm.amdgcn.mfma.f32.4x4x1f32(float %{{.*}}, float %{{.*}}, <4 x float> %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  %r3 = rocdl.mfma.f32.4x4x1f32 %arg0, %arg1, %arg5, 0, 0, 0 :
                            (f32, f32, vector<4xf32>) -> vector<4xf32>

  // CHECK: call <16 x float> @llvm.amdgcn.mfma.f32.32x32x2f32(float %{{.*}}, float %{{.*}}, <16 x float> %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  %r4= rocdl.mfma.f32.32x32x2f32 %arg0, %arg1, %arg4, 0, 0, 0 :
                            (f32, f32, vector<16 x f32>) -> vector<16 x f32>

  // CHECK: call <32 x float> @llvm.amdgcn.mfma.f32.32x32x4f16(<4 x half> %{{.*}}, <4 x half> %{{.*}}, <32 x float> %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  %r5 = rocdl.mfma.f32.32x32x4f16 %arg6, %arg6, %arg2, 0, 0, 0 :
                            (vector<4xf16>, vector<4xf16>, vector<32 x f32>) -> vector<32 x f32>

  // CHECK: call <16 x float> @llvm.amdgcn.mfma.f32.16x16x4f16(<4 x half> %{{.*}}, <4 x half> %{{.*}}, <16 x float> %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  %r6 = rocdl.mfma.f32.16x16x4f16 %arg6, %arg6, %arg4, 0, 0, 0 :
                            (vector<4xf16>, vector<4xf16>, vector<16 x f32>) -> vector<16 x f32>

  // CHECK: call <4 x float> @llvm.amdgcn.mfma.f32.4x4x4f16(<4 x half> %{{.*}}, <4 x half> %{{.*}}, <4 x float> %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  %r7 = rocdl.mfma.f32.4x4x4f16 %arg6, %arg6, %arg5, 0, 0, 0 :
                            (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>

  // CHECK: call <16 x float> @llvm.amdgcn.mfma.f32.32x32x8f16(<4 x half> %{{.*}}, <4 x half> %{{.*}}, <16 x float> %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  %r8 = rocdl.mfma.f32.32x32x8f16 %arg6, %arg6, %arg4, 0, 0, 0 :
                            (vector<4xf16>, vector<4xf16>, vector<16 x f32>) -> vector<16 x f32>

  // CHECK: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half> %{{.*}}, <4 x half> %{{.*}}, <4 x float> %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  %r9 = rocdl.mfma.f32.16x16x16f16 %arg6, %arg6, %arg5, 0, 0, 0 :
                            (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>

  // CHECK: call <32 x i32> @llvm.amdgcn.mfma.i32.32x32x4i8(i32 %{{.*}}, i32 %{{.*}}, <32 x i32> %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  %r10 = rocdl.mfma.i32.32x32x4i8 %arg3, %arg3, %arg7, 0, 0, 0 :
                            (i32, i32, vector<32 x i32>) -> vector<32 x i32>

  // CHECK: call <16 x i32> @llvm.amdgcn.mfma.i32.16x16x4i8(i32 %{{.*}}, i32 %{{.*}}, <16 x i32> %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  %r11 = rocdl.mfma.i32.16x16x4i8 %arg3, %arg3, %arg8, 0, 0, 0 :
                            (i32, i32, vector<16 x i32>) -> vector<16 x i32>

  // CHECK: call <4 x i32> @llvm.amdgcn.mfma.i32.4x4x4i8(i32 %{{.*}}, i32 %{{.*}}, <4 x i32> %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  %r12 = rocdl.mfma.i32.4x4x4i8 %arg3, %arg3, %arg9, 0, 0, 0 :
                            (i32, i32, vector<4xi32>) -> vector<4xi32>

  // CHECK: call <16 x i32> @llvm.amdgcn.mfma.i32.32x32x8i8(i32 %{{.*}}, i32 %{{.*}}, <16 x i32> %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  %r13 = rocdl.mfma.i32.32x32x8i8 %arg3, %arg3, %arg8, 0, 0, 0 :
                            (i32, i32, vector<16 x i32>) -> vector<16 x i32>

  // CHECK: call <4 x i32> @llvm.amdgcn.mfma.i32.16x16x16i8(i32 %{{.*}}, i32 %{{.*}}, <4 x i32> %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  %r14 = rocdl.mfma.i32.16x16x16i8 %arg3, %arg3, %arg9, 0, 0, 0 :
                            (i32, i32, vector<4xi32>) -> vector<4xi32>

  // CHECK: call <32 x float> @llvm.amdgcn.mfma.f32.32x32x2bf16(<2 x i16> %{{.*}}, <2 x i16> %{{.*}}, <32 x float> %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  %r15 = rocdl.mfma.f32.32x32x2bf16 %arg10, %arg10, %arg2, 0, 0, 0 :
                            (vector<2xi16>, vector<2xi16>, vector<32 x f32>) -> vector<32 x f32>

  // CHECK: call <16 x float> @llvm.amdgcn.mfma.f32.16x16x2bf16(<2 x i16> %{{.*}}, <2 x i16> %{{.*}}, <16 x float> %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  %r16 = rocdl.mfma.f32.16x16x2bf16 %arg10, %arg10, %arg4, 0, 0, 0 :
                            (vector<2xi16>, vector<2xi16>, vector<16 x f32>) -> vector<16 x f32>

  // CHECK: call <4 x float> @llvm.amdgcn.mfma.f32.4x4x2bf16(<2 x i16> %{{.*}}, <2 x i16> %{{.*}}, <4 x float> %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  %r17 = rocdl.mfma.f32.4x4x2bf16 %arg10, %arg10, %arg5, 0, 0, 0 :
                            (vector<2xi16>, vector<2xi16>, vector<4xf32>) -> vector<4xf32>

  // CHECK: call <16 x float> @llvm.amdgcn.mfma.f32.32x32x4bf16(<2 x i16> %{{.*}}, <2 x i16> %{{.*}}, <16 x float> %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  %r18 = rocdl.mfma.f32.32x32x4bf16 %arg10, %arg10, %arg4, 0, 0, 0 :
                            (vector<2xi16>, vector<2xi16>, vector<16 x f32>) -> vector<16 x f32>

  // CHECK: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x8bf16(<2 x i16> %{{.*}}, <2 x i16> %{{.*}}, <4 x float> %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  %r19 = rocdl.mfma.f32.16x16x8bf16 %arg10, %arg10, %arg5, 0, 0, 0 :
                            (vector<2xi16>, vector<2xi16>, vector<4xf32>) -> vector<4xf32>

  // CHECK: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf8.bf8(i64 %{{.*}}, i64 %{{.*}}, <4 x float> %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  %r20 = rocdl.mfma.f32.16x16x32.bf8.bf8 %arg11, %arg11, %arg5, 0, 0, 0 :
                            (i64, i64, vector<4xf32>) -> vector<4xf32>

  // CHECK: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf8.fp8(i64 %{{.*}}, i64 %{{.*}}, <4 x float> %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  %r21 = rocdl.mfma.f32.16x16x32.bf8.fp8 %arg11, %arg11, %arg5, 0, 0, 0 :
                            (i64, i64, vector<4xf32>) -> vector<4xf32>

  // CHECK: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.bf8(i64 %{{.*}}, i64 %{{.*}}, <4 x float> %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  %r22 = rocdl.mfma.f32.16x16x32.fp8.bf8 %arg11, %arg11, %arg5, 0, 0, 0 :
                            (i64, i64, vector<4xf32>) -> vector<4xf32>

  // CHECK: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 %{{.*}}, i64 %{{.*}}, <4 x float> %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  %r23 = rocdl.mfma.f32.16x16x32.fp8.fp8 %arg11, %arg11, %arg5, 0, 0, 0 :
                            (i64, i64, vector<4xf32>) -> vector<4xf32>

  // CHECK: call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf8.bf8(i64 %{{.*}}, i64 %{{.*}}, <16 x float> %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  %r24 = rocdl.mfma.f32.32x32x16.bf8.bf8 %arg11, %arg11, %arg4, 0, 0, 0 :
                            (i64, i64, vector<16xf32>) -> vector<16xf32>

  // CHECK: call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf8.fp8(i64 %{{.*}}, i64 %{{.*}}, <16 x float> %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  %r25 = rocdl.mfma.f32.32x32x16.bf8.fp8 %arg11, %arg11, %arg4, 0, 0, 0 :
                            (i64, i64, vector<16xf32>) -> vector<16xf32>

  // CHECK: call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.fp8.bf8(i64 %{{.*}}, i64 %{{.*}}, <16 x float> %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  %r26 = rocdl.mfma.f32.32x32x16.fp8.bf8 %arg11, %arg11, %arg4, 0, 0, 0 :
                            (i64, i64, vector<16xf32>) -> vector<16xf32>

  // CHECK: call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf8.bf8(i64 %{{.*}}, i64 %{{.*}}, <16 x float> %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  %r27 = rocdl.mfma.f32.32x32x16.bf8.bf8 %arg11, %arg11, %arg4, 0, 0, 0 :
                            (i64, i64, vector<16xf32>) -> vector<16xf32>

  // CHECK: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}}, <4 x float> %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  %r28 = rocdl.mfma.f32.16x16x32.bf16 %arg12, %arg12, %arg5, 0, 0, 0 :
                              (vector<8xbf16>, vector<8xbf16>, vector<4xf32>) -> vector<4xf32>

  // CHECK: call <4 x i32> @llvm.amdgcn.mfma.i32.16x16x64.i8(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  %r29 = rocdl.mfma.i32.16x16x64.i8 %arg9, %arg9, %arg9, 0, 0, 0 :
                              (vector<4xi32>, vector<4xi32>, vector<4xi32>) -> vector<4xi32>

  // CHECK: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %{{.*}}, <8 x half> %{{.*}}, <4 x float> %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  %r30 = rocdl.mfma.f32.16x16x32.f16 %arg14, %arg14, %arg5, 0, 0, 0 :
                               (vector<8xf16>, vector<8xf16>, vector<4xf32>) -> vector<4xf32>

  // CHECK: call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> %1{{.*}}, <8 x bfloat> %{{.*}}, <16 x float> %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  %r31 = rocdl.mfma.f32.32x32x16.bf16 %arg12, %arg12, %arg4, 0, 0, 0 :
                               (vector<8xbf16>, vector<8xbf16>, vector<16xf32>) -> vector<16xf32>

  // CHECK: call <16 x i32> @llvm.amdgcn.mfma.i32.32x32x32.i8(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <16 x i32> %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  %r32 = rocdl.mfma.i32.32x32x32.i8 %arg9, %arg9, %arg8, 0, 0, 0 :
                               (vector<4xi32>, vector<4xi32>, vector<16xi32>) -> vector<16xi32>

  // CHECK: call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %{{.*}}, <8 x half> %{{.*}}, <16 x float> %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  %r33 = rocdl.mfma.f32.32x32x16.f16 %arg14, %arg14, %arg4, 0, 0, 0 :
                               (vector<8xf16>, vector<8xf16>, vector<16xf32>) -> vector<16xf32>

  llvm.return %r0 : vector<32 x f32>
}

llvm.func @rocdl.smfmac(%arg0 : i32,
                   %arg1 : vector<4 x f16>,
                   %arg2 : vector<8 x f16>,
                   %arg3 : vector<4 x f32>,
                   %arg4 : vector<16 x f32>,
                   %arg5 : vector<4 x i16>,
                   %arg6 : vector<8 x i16>,
                   %arg7 : vector<2xi32>,
                   %arg8 : vector<4xi32>,
                   %arg9 : vector<16xi32>,
                   %arg10 : vector<16 x f16>,
                   %arg11 : vector<8 x bf16>,
                   %arg12 : vector<16 x bf16>,
                   %arg13 : vector<8 x i32>) -> vector<4 x f32> {
  %index = llvm.mlir.constant(42 : i32) : i32

  // CHECK-LABEL: rocdl.smfmac

  // CHECK: call <4 x float> @llvm.amdgcn.smfmac.f32.16x16x32.f16(<4 x half> %{{.*}}, <8 x half> %{{.*}}, <4 x float> %{{.*}}, i32 42, i32 0, i32 0)
  %r0 = rocdl.smfmac.f32.16x16x32.f16 %arg1, %arg2, %arg3, %index, 0, 0 :
                                (vector<4xf16>, vector<8xf16>, vector<4xf32>, i32) -> vector<4xf32>

  // CHECK: call <16 x float> @llvm.amdgcn.smfmac.f32.32x32x16.f16(<4 x half> %{{.*}}, <8 x half> %{{.*}}, <16 x float> %{{.*}}, i32 42, i32 0, i32 0)
  %r1 = rocdl.smfmac.f32.32x32x16.f16 %arg1, %arg2, %arg4, %index, 0, 0 :
                                (vector<4xf16>, vector<8xf16>, vector<16xf32>, i32) -> vector<16xf32>

  // CHECK: call <4 x float> @llvm.amdgcn.smfmac.f32.16x16x32.bf16(<4 x i16> %{{.*}}, <8 x i16> %{{.*}}, <4 x float> %{{.*}}, i32 42, i32 0, i32 0)
  %r2 = rocdl.smfmac.f32.16x16x32.bf16 %arg5, %arg6, %arg3, %index, 0, 0 :
                                (vector<4xi16>, vector<8xi16>, vector<4xf32>, i32) -> vector<4xf32>

  // CHECK: call <16 x float> @llvm.amdgcn.smfmac.f32.32x32x16.bf16(<4 x i16> %{{.*}}, <8 x i16> %{{.*}}, <16 x float> %{{.*}}, i32 42, i32 0, i32 0)
  %r3 = rocdl.smfmac.f32.32x32x16.bf16 %arg5, %arg6, %arg4, %index, 0, 0 :
                                (vector<4xi16>, vector<8xi16>, vector<16xf32>, i32) -> vector<16xf32>

  // CHECK: call <4 x i32> @llvm.amdgcn.smfmac.i32.16x16x64.i8(<2 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 42, i32 0, i32 0)
  %r4 = rocdl.smfmac.i32.16x16x64.i8 %arg7, %arg8, %arg8, %index, 0, 0 :
                                (vector<2xi32>, vector<4xi32>, vector<4xi32>, i32) -> vector<4xi32>

  // CHECK: call <16 x i32> @llvm.amdgcn.smfmac.i32.32x32x32.i8(<2 x i32> %{{.*}}, <4 x i32> %{{.*}}, <16 x i32> %{{.*}}, i32 42, i32 0, i32 0)
  %r5 = rocdl.smfmac.i32.32x32x32.i8 %arg7, %arg8, %arg9, %index, 0, 0 :
                                (vector<2xi32>, vector<4xi32>, vector<16xi32>, i32) -> vector<16xi32>

  // CHECK: call <4 x float> @llvm.amdgcn.smfmac.f32.16x16x64.bf8.bf8(<2 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x float> %{{.*}}, i32 42, i32 0, i32 0)
  %r6 = rocdl.smfmac.f32.16x16x64.bf8.bf8 %arg7, %arg8, %arg3, %index, 0, 0 :
                                (vector<2xi32>, vector<4xi32>, vector<4xf32>, i32) -> vector<4xf32>

  // CHECK: call <4 x float> @llvm.amdgcn.smfmac.f32.16x16x64.bf8.fp8(<2 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x float> %{{.*}}, i32 42, i32 0, i32 0)
  %r7 = rocdl.smfmac.f32.16x16x64.bf8.fp8 %arg7, %arg8, %arg3, %index, 0, 0 :
                                (vector<2xi32>, vector<4xi32>, vector<4xf32>, i32) -> vector<4xf32>

  // CHECK: call <4 x float> @llvm.amdgcn.smfmac.f32.16x16x64.fp8.bf8(<2 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x float> %{{.*}}, i32 42, i32 0, i32 0)
  %r8 = rocdl.smfmac.f32.16x16x64.fp8.bf8 %arg7, %arg8, %arg3, %index, 0, 0 :
                                (vector<2xi32>, vector<4xi32>, vector<4xf32>, i32) -> vector<4xf32>

  // CHECK: call <4 x float> @llvm.amdgcn.smfmac.f32.16x16x64.fp8.fp8(<2 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x float> %{{.*}}, i32 42, i32 0, i32 0)
  %r9 = rocdl.smfmac.f32.16x16x64.fp8.fp8 %arg7, %arg8, %arg3, %index, 0, 0 :
                                (vector<2xi32>, vector<4xi32>, vector<4xf32>, i32) -> vector<4xf32>

  // CHECK: call <16 x float> @llvm.amdgcn.smfmac.f32.32x32x32.bf8.bf8(<2 x i32> %{{.*}}, <4 x i32> %{{.*}}, <16 x float> %{{.*}}, i32 42, i32 0, i32 0)
  %r10 = rocdl.smfmac.f32.32x32x32.bf8.bf8 %arg7, %arg8, %arg4, %index, 0, 0 :
                                (vector<2xi32>, vector<4xi32>, vector<16xf32>, i32) -> vector<16xf32>

  // CHECK: call <16 x float> @llvm.amdgcn.smfmac.f32.32x32x32.bf8.fp8(<2 x i32> %{{.*}}, <4 x i32> %{{.*}}, <16 x float> %{{.*}}, i32 42, i32 0, i32 0)
  %r11 = rocdl.smfmac.f32.32x32x32.bf8.fp8 %arg7, %arg8, %arg4, %index, 0, 0 :
                                (vector<2xi32>, vector<4xi32>, vector<16xf32>, i32) -> vector<16xf32>

  // CHECK: call <16 x float> @llvm.amdgcn.smfmac.f32.32x32x32.fp8.bf8(<2 x i32> %{{.*}}, <4 x i32> %{{.*}}, <16 x float> %{{.*}}, i32 42, i32 0, i32 0)
  %r12 = rocdl.smfmac.f32.32x32x32.fp8.bf8 %arg7, %arg8, %arg4, %index, 0, 0 :
                                (vector<2xi32>, vector<4xi32>, vector<16xf32>, i32) -> vector<16xf32>

  // CHECK: call <16 x float> @llvm.amdgcn.smfmac.f32.32x32x32.fp8.fp8(<2 x i32> %{{.*}}, <4 x i32> %{{.*}}, <16 x float> %{{.*}}, i32 42, i32 0, i32 0)
  %r13 = rocdl.smfmac.f32.32x32x32.fp8.fp8 %arg7, %arg8, %arg4, %index, 0, 0 :
                                (vector<2xi32>, vector<4xi32>, vector<16xf32>, i32) -> vector<16xf32>

  // CHECK: call <4 x float> @llvm.amdgcn.smfmac.f32.16x16x64.f16(<8 x half> %{{.*}}, <16 x half> %{{.*}}, <4 x float> %{{.*}}, i32 42, i32 0, i32 0)
  %r14 = rocdl.smfmac.f32.16x16x64.f16 %arg2, %arg10, %arg3, %index, 0, 0 :
                                (vector<8xf16>, vector<16xf16>, vector<4xf32>, i32) -> vector<4xf32>

  // CHECK: call <16 x float> @llvm.amdgcn.smfmac.f32.32x32x32.f16(<8 x half> %{{.*}}, <16 x half> %{{.*}}, <16 x float> %{{.*}}, i32 42, i32 0, i32 0)
  %r15 = rocdl.smfmac.f32.32x32x32.f16 %arg2, %arg10, %arg4, %index, 0, 0 :
                                (vector<8xf16>, vector<16xf16>, vector<16xf32>, i32) -> vector<16xf32>

  // CHECK: call <4 x float> @llvm.amdgcn.smfmac.f32.16x16x64.bf16(<8 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}}, <4 x float> %{{.*}}, i32 42, i32 0, i32 0)
  %r16 = rocdl.smfmac.f32.16x16x64.bf16 %arg11, %arg12, %arg3, %index, 0, 0 :
                                (vector<8xbf16>, vector<16xbf16>, vector<4xf32>, i32) -> vector<4xf32>

  // CHECK: call <16 x float> @llvm.amdgcn.smfmac.f32.32x32x32.bf16(<8 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}}, <16 x float> %{{.*}}, i32 42, i32 0, i32 0)
  %r17 = rocdl.smfmac.f32.32x32x32.bf16 %arg11, %arg12, %arg4, %index, 0, 0 :
                                (vector<8xbf16>, vector<16xbf16>, vector<16xf32>, i32) -> vector<16xf32>

  // CHECK: call <4 x i32> @llvm.amdgcn.smfmac.i32.16x16x128.i8(<4 x i32> %{{.*}}, <8 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 42, i32 0, i32 0)
  %r18 = rocdl.smfmac.i32.16x16x128.i8 %arg8, %arg13, %arg8, %index, 0, 0 :
                                (vector<4xi32>, vector<8xi32>, vector<4xi32>, i32) -> vector<4xi32>

  // CHECK: call <4 x float> @llvm.amdgcn.smfmac.f32.16x16x128.bf8.bf8(<4 x i32> %{{.*}}, <8 x i32> %{{.*}}, <4 x float> %{{.*}}, i32 42, i32 0, i32 0)
  %r19 = rocdl.smfmac.f32.16x16x128.bf8.bf8 %arg8, %arg13, %arg3, %index, 0, 0 :
                                (vector<4xi32>, vector<8xi32>, vector<4xf32>, i32) -> vector<4xf32>

  // CHECK: call <4 x float> @llvm.amdgcn.smfmac.f32.16x16x128.bf8.fp8(<4 x i32> %{{.*}}, <8 x i32> %{{.*}}, <4 x float> %{{.*}}, i32 42, i32 0, i32 0)
  %r20 = rocdl.smfmac.f32.16x16x128.bf8.fp8 %arg8, %arg13, %arg3, %index, 0, 0 :
                                (vector<4xi32>, vector<8xi32>, vector<4xf32>, i32) -> vector<4xf32>

  // CHECK: call <4 x float> @llvm.amdgcn.smfmac.f32.16x16x128.fp8.bf8(<4 x i32> %{{.*}}, <8 x i32> %{{.*}}, <4 x float> %{{.*}}, i32 42, i32 0, i32 0)
  %r21 = rocdl.smfmac.f32.16x16x128.fp8.bf8 %arg8, %arg13, %arg3, %index, 0, 0 :
                                (vector<4xi32>, vector<8xi32>, vector<4xf32>, i32) -> vector<4xf32>

  // CHECK: call <4 x float> @llvm.amdgcn.smfmac.f32.16x16x128.fp8.fp8(<4 x i32> %{{.*}}, <8 x i32> %{{.*}}, <4 x float> %{{.*}}, i32 42, i32 0, i32 0)
  %r22 = rocdl.smfmac.f32.16x16x128.fp8.fp8 %arg8, %arg13, %arg3, %index, 0, 0 :
                                (vector<4xi32>, vector<8xi32>, vector<4xf32>, i32) -> vector<4xf32>

  // CHECK: call <16 x i32> @llvm.amdgcn.smfmac.i32.32x32x64.i8(<4 x i32> %{{.*}}, <8 x i32> %{{.*}}, <16 x i32> %{{.*}}, i32 42, i32 0, i32 0)
  %r23 = rocdl.smfmac.i32.32x32x64.i8 %arg8, %arg13, %arg9, %index, 0, 0 :
                                (vector<4xi32>, vector<8xi32>, vector<16xi32>, i32) -> vector<16xi32>

  // CHECK: call <16 x float> @llvm.amdgcn.smfmac.f32.32x32x64.bf8.bf8(<4 x i32> %{{.*}}, <8 x i32> %{{.*}}, <16 x float> %{{.*}}, i32 42, i32 0, i32 0)
  %r24 = rocdl.smfmac.f32.32x32x64.bf8.bf8 %arg8, %arg13, %arg4, %index, 0, 0 :
                                (vector<4xi32>, vector<8xi32>, vector<16xf32>, i32) -> vector<16xf32>

  // CHECK: call <16 x float> @llvm.amdgcn.smfmac.f32.32x32x64.bf8.fp8(<4 x i32> %{{.*}}, <8 x i32> %{{.*}}, <16 x float> %{{.*}}, i32 42, i32 0, i32 0)
  %r25 = rocdl.smfmac.f32.32x32x64.bf8.fp8 %arg8, %arg13, %arg4, %index, 0, 0 :
                                (vector<4xi32>, vector<8xi32>, vector<16xf32>, i32) -> vector<16xf32>

  // CHECK: call <16 x float> @llvm.amdgcn.smfmac.f32.32x32x64.fp8.bf8(<4 x i32> %{{.*}}, <8 x i32> %{{.*}}, <16 x float> %{{.*}}, i32 42, i32 0, i32 0)
  %r26 = rocdl.smfmac.f32.32x32x64.fp8.bf8 %arg8, %arg13, %arg4, %index, 0, 0 :
                                (vector<4xi32>, vector<8xi32>, vector<16xf32>, i32) -> vector<16xf32>

  // CHECK: call <16 x float> @llvm.amdgcn.smfmac.f32.32x32x64.fp8.fp8(<4 x i32> %{{.*}}, <8 x i32> %{{.*}}, <16 x float> %{{.*}}, i32 42, i32 0, i32 0)
  %r27 = rocdl.smfmac.f32.32x32x64.fp8.fp8 %arg8, %arg13, %arg4, %index, 0, 0 :
                                (vector<4xi32>, vector<8xi32>, vector<16xf32>, i32) -> vector<16xf32>

  llvm.return %r0 : vector<4 x f32>
}


llvm.func @rocdl.mfma.scale.f32.32x32x64.f8f6f4(%arg0 : i32,
                   %arg1 : vector<16 x f32>, %arg2 : vector<8xi32>,
                   %arg3 : vector<6xi32>, %arg4 : vector<4xi32>) -> vector<16 x f32> {

  // CHECK-LABEL: rocdl.mfma.scale.f32.32x32x64.f8f6f4
  // fp8 * fp8
  // CHECK: call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32(<8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <16 x float> %{{.*}}, i32 0, i32 0, i32 0, i32 %{{.*}}, i32 0, i32 %{{.*}})
  %r00 = rocdl.mfma.scale.f32.32x32x64.f8f6f4 %arg2, %arg2, %arg1, 0, 0, 0, %arg0, 0, %arg0 :
                              (vector<8xi32>, vector<8xi32>, vector<16xf32>, i32, i32) -> vector<16xf32>

  // fp8 * bf8
  // CHECK: call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32(<8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <16 x float> %{{.*}}, i32 0, i32 1, i32 0, i32 %{{.*}}, i32 0, i32 %{{.*}})
  %r01 = rocdl.mfma.scale.f32.32x32x64.f8f6f4 %arg2, %arg2, %arg1, 0, 1, 0, %arg0, 0, %arg0 :
                              (vector<8xi32>, vector<8xi32>, vector<16xf32>, i32, i32) -> vector<16xf32>

  // fp8 * fp6
  // CHECK: call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v6i32(<8 x i32> %{{.*}}, <6 x i32> %{{.*}}, <16 x float> %{{.*}}, i32 0, i32 2, i32 0, i32 %{{.*}}, i32 0, i32 %{{.*}})
  %r02 = rocdl.mfma.scale.f32.32x32x64.f8f6f4 %arg2, %arg3, %arg1, 0, 2, 0, %arg0, 0, %arg0 :
                              (vector<8xi32>, vector<6xi32>, vector<16xf32>, i32, i32) -> vector<16xf32>

  // fp8 * bf6
  // CHECK: call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v6i32(<8 x i32> %{{.*}}, <6 x i32> %{{.*}}, <16 x float> %{{.*}}, i32 0, i32 3, i32 0, i32 %{{.*}}, i32 0, i32 %{{.*}})
  %r03 = rocdl.mfma.scale.f32.32x32x64.f8f6f4 %arg2, %arg3, %arg1, 0, 3, 0, %arg0, 0, %arg0 :
                              (vector<8xi32>, vector<6xi32>, vector<16xf32>, i32, i32) -> vector<16xf32>

  // fp8 * fp4
  // CHECK: call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v4i32(<8 x i32> %{{.*}}, <4 x i32> %{{.*}}, <16 x float> %{{.*}}, i32 0, i32 4, i32 0, i32 %{{.*}}, i32 0, i32 %{{.*}})
  %r04 = rocdl.mfma.scale.f32.32x32x64.f8f6f4 %arg2, %arg4, %arg1, 0, 4, 0, %arg0, 0, %arg0 :
                              (vector<8xi32>, vector<4xi32>, vector<16xf32>, i32, i32) -> vector<16xf32>

  // bf8 * fp8
  // CHECK: call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32(<8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <16 x float> %{{.*}}, i32 1, i32 0, i32 0, i32 %{{.*}}, i32 0, i32 %{{.*}})
  %r10 = rocdl.mfma.scale.f32.32x32x64.f8f6f4 %arg2, %arg2, %arg1, 1, 0, 0, %arg0, 0, %arg0 :
                              (vector<8xi32>, vector<8xi32>, vector<16xf32>, i32, i32) -> vector<16xf32>

  // bf8 * bf8
  // CHECK: call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32(<8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <16 x float> %{{.*}}, i32 1, i32 1, i32 0, i32 %{{.*}}, i32 0, i32 %{{.*}})
  %r11 = rocdl.mfma.scale.f32.32x32x64.f8f6f4 %arg2, %arg2, %arg1, 1, 1, 0, %arg0, 0, %arg0 :
                              (vector<8xi32>, vector<8xi32>, vector<16xf32>, i32, i32) -> vector<16xf32>

  // bf8 * fp6
  // CHECK: call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v6i32(<8 x i32> %{{.*}}, <6 x i32> %{{.*}}, <16 x float> %{{.*}}, i32 1, i32 2, i32 0, i32 %{{.*}}, i32 0, i32 %{{.*}})
  %r12 = rocdl.mfma.scale.f32.32x32x64.f8f6f4 %arg2, %arg3, %arg1, 1, 2, 0, %arg0, 0, %arg0 :
                              (vector<8xi32>, vector<6xi32>, vector<16xf32>, i32, i32) -> vector<16xf32>

  // bf8 * bf6
  // CHECK: call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v6i32(<8 x i32> %{{.*}}, <6 x i32> %{{.*}}, <16 x float> %{{.*}}, i32 1, i32 3, i32 0, i32 %{{.*}}, i32 0, i32 %{{.*}})
  %r13 = rocdl.mfma.scale.f32.32x32x64.f8f6f4 %arg2, %arg3, %arg1, 1, 3, 0, %arg0, 0, %arg0 :
                              (vector<8xi32>, vector<6xi32>, vector<16xf32>, i32, i32) -> vector<16xf32>

  // bf8 * fp4
  // CHECK: call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v4i32(<8 x i32> %{{.*}}, <4 x i32> %{{.*}}, <16 x float> %{{.*}}, i32 1, i32 4, i32 0, i32 %{{.*}}, i32 0, i32 %{{.*}})
  %r14 = rocdl.mfma.scale.f32.32x32x64.f8f6f4 %arg2, %arg4, %arg1, 1, 4, 0, %arg0, 0, %arg0 :
                              (vector<8xi32>, vector<4xi32>, vector<16xf32>, i32, i32) -> vector<16xf32>

  // fp6 * fp8
  // CHECK: call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v6i32.v8i32(<6 x i32> %{{.*}}, <8 x i32> %{{.*}}, <16 x float> %{{.*}}, i32 2, i32 0, i32 0, i32 %{{.*}}, i32 0, i32 %{{.*}})
  %r20 = rocdl.mfma.scale.f32.32x32x64.f8f6f4 %arg3, %arg2, %arg1, 2, 0, 0, %arg0, 0, %arg0 :
                              (vector<6xi32>, vector<8xi32>, vector<16xf32>, i32, i32) -> vector<16xf32>

  // fp6 * bf8
  // CHECK: call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v6i32.v8i32(<6 x i32> %{{.*}}, <8 x i32> %{{.*}}, <16 x float> %{{.*}}, i32 2, i32 1, i32 0, i32 %{{.*}}, i32 0, i32 %{{.*}})
  %r21 = rocdl.mfma.scale.f32.32x32x64.f8f6f4 %arg3, %arg2, %arg1, 2, 1, 0, %arg0, 0, %arg0 :
                              (vector<6xi32>, vector<8xi32>, vector<16xf32>, i32, i32) -> vector<16xf32>

  // fp6 * fp6
  // CHECK: call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v6i32.v6i32(<6 x i32> %{{.*}}, <6 x i32> %{{.*}}, <16 x float> %{{.*}}, i32 2, i32 2, i32 0, i32 %{{.*}}, i32 0, i32 %{{.*}})
  %r22 = rocdl.mfma.scale.f32.32x32x64.f8f6f4 %arg3, %arg3, %arg1, 2, 2, 0, %arg0, 0, %arg0 :
                              (vector<6xi32>, vector<6xi32>, vector<16xf32>, i32, i32) -> vector<16xf32>

  // fp6 * bf6
  // CHECK: call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v6i32.v6i32(<6 x i32> %{{.*}}, <6 x i32> %{{.*}}, <16 x float> %{{.*}}, i32 2, i32 3, i32 0, i32 %{{.*}}, i32 0, i32 %{{.*}})
  %r23 = rocdl.mfma.scale.f32.32x32x64.f8f6f4 %arg3, %arg3, %arg1, 2, 3, 0, %arg0, 0, %arg0 :
                              (vector<6xi32>, vector<6xi32>, vector<16xf32>, i32, i32) -> vector<16xf32>

  // fp6 * fp4
  // CHECK: call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v6i32.v4i32(<6 x i32> %{{.*}}, <4 x i32> %{{.*}}, <16 x float> %{{.*}}, i32 2, i32 4, i32 0, i32 %{{.*}}, i32 0, i32 %{{.*}})
  %r24 = rocdl.mfma.scale.f32.32x32x64.f8f6f4 %arg3, %arg4, %arg1, 2, 4, 0, %arg0, 0, %arg0 :
                              (vector<6xi32>, vector<4xi32>, vector<16xf32>, i32, i32) -> vector<16xf32>

  // bf6 * fp8
  // CHECK: call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v6i32.v8i32(<6 x i32> %{{.*}}, <8 x i32> %{{.*}}, <16 x float> %{{.*}}, i32 3, i32 0, i32 0, i32 %{{.*}}, i32 0, i32 %{{.*}})
  %r30 = rocdl.mfma.scale.f32.32x32x64.f8f6f4 %arg3, %arg2, %arg1, 3, 0, 0, %arg0, 0, %arg0 :
                              (vector<6xi32>, vector<8xi32>, vector<16xf32>, i32, i32) -> vector<16xf32>

  // bf6 * bf8
  // CHECK: call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v6i32.v8i32(<6 x i32> %{{.*}}, <8 x i32> %{{.*}}, <16 x float> %{{.*}}, i32 3, i32 1, i32 0, i32 %{{.*}}, i32 0, i32 %{{.*}})
  %r31 = rocdl.mfma.scale.f32.32x32x64.f8f6f4 %arg3, %arg2, %arg1, 3, 1, 0, %arg0, 0, %arg0 :
                              (vector<6xi32>, vector<8xi32>, vector<16xf32>, i32, i32) -> vector<16xf32>

  // bf6 * fp6
  // CHECK: call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v6i32.v6i32(<6 x i32> %{{.*}}, <6 x i32> %{{.*}}, <16 x float> %{{.*}}, i32 3, i32 2, i32 0, i32 %{{.*}}, i32 0, i32 %{{.*}})
  %r32 = rocdl.mfma.scale.f32.32x32x64.f8f6f4 %arg3, %arg3, %arg1, 3, 2, 0, %arg0, 0, %arg0 :
                              (vector<6xi32>, vector<6xi32>, vector<16xf32>, i32, i32) -> vector<16xf32>

  // bf6 * bf6
  // CHECK: call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v6i32.v6i32(<6 x i32> %{{.*}}, <6 x i32> %{{.*}}, <16 x float> %{{.*}}, i32 3, i32 3, i32 0, i32 %{{.*}}, i32 0, i32 %{{.*}})
  %r33 = rocdl.mfma.scale.f32.32x32x64.f8f6f4 %arg3, %arg3, %arg1, 3, 3, 0, %arg0, 0, %arg0 :
                              (vector<6xi32>, vector<6xi32>, vector<16xf32>, i32, i32) -> vector<16xf32>

  // bf6 * fp4
  // CHECK: call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v6i32.v4i32(<6 x i32> %{{.*}}, <4 x i32> %{{.*}}, <16 x float> %{{.*}}, i32 3, i32 4, i32 0, i32 %{{.*}}, i32 0, i32 %{{.*}})
  %r34 = rocdl.mfma.scale.f32.32x32x64.f8f6f4 %arg3, %arg4, %arg1, 3, 4, 0, %arg0, 0, %arg0 :
                              (vector<6xi32>, vector<4xi32>, vector<16xf32>, i32, i32) -> vector<16xf32>

  // fp4 * fp8
  // CHECK: call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v4i32.v8i32(<4 x i32> %{{.*}}, <8 x i32> %{{.*}}, <16 x float> %{{.*}}, i32 4, i32 0, i32 0, i32 %{{.*}}, i32 0, i32 %{{.*}})
  %r40 = rocdl.mfma.scale.f32.32x32x64.f8f6f4 %arg4, %arg2, %arg1, 4, 0, 0, %arg0, 0, %arg0 :
                              (vector<4xi32>, vector<8xi32>, vector<16xf32>, i32, i32) -> vector<16xf32>

  // fp4 * bf8
  // CHECK: call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v4i32.v8i32(<4 x i32> %{{.*}}, <8 x i32> %{{.*}}, <16 x float> %{{.*}}, i32 4, i32 1, i32 0, i32 %{{.*}}, i32 0, i32 %{{.*}})
  %r41 = rocdl.mfma.scale.f32.32x32x64.f8f6f4 %arg4, %arg2, %arg1, 4, 1, 0, %arg0, 0, %arg0 :
                              (vector<4xi32>, vector<8xi32>, vector<16xf32>, i32, i32) -> vector<16xf32>

  // fp4 * fp6
  // CHECK: call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v4i32.v6i32(<4 x i32> %{{.*}}, <6 x i32> %{{.*}}, <16 x float> %{{.*}}, i32 4, i32 2, i32 0, i32 %{{.*}}, i32 0, i32 %{{.*}})
  %r42 = rocdl.mfma.scale.f32.32x32x64.f8f6f4 %arg4, %arg3, %arg1, 4, 2, 0, %arg0, 0, %arg0 :
                              (vector<4xi32>, vector<6xi32>, vector<16xf32>, i32, i32) -> vector<16xf32>

  // fp4 * bf6
  // CHECK: call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v4i32.v6i32(<4 x i32> %{{.*}}, <6 x i32> %{{.*}}, <16 x float> %{{.*}}, i32 4, i32 3, i32 0, i32 %{{.*}}, i32 0, i32 %{{.*}})
  %r43 = rocdl.mfma.scale.f32.32x32x64.f8f6f4 %arg4, %arg3, %arg1, 4, 3, 0, %arg0, 0, %arg0 :
                              (vector<4xi32>, vector<6xi32>, vector<16xf32>, i32, i32) -> vector<16xf32>

  // fp4 * fp4
  // CHECK: call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v4i32.v4i32(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <16 x float> %{{.*}}, i32 4, i32 4, i32 0, i32 %{{.*}}, i32 0, i32 %{{.*}})
  %r44 = rocdl.mfma.scale.f32.32x32x64.f8f6f4 %arg4, %arg4, %arg1, 4, 4, 0, %arg0, 0, %arg0 :
                              (vector<4xi32>, vector<4xi32>, vector<16xf32>, i32, i32) -> vector<16xf32>

  llvm.return %r00 : vector<16 x f32>
}

llvm.func @rocdl.mfma.scale.f32.16x16x128.f8f6f4(%arg0 : i32,
                   %arg1 : vector<4 x f32>, %arg2 : vector<8xi32>,
                   %arg3 : vector<6xi32>, %arg4 : vector<4xi32>) -> vector<4 x f32> {

  // CHECK-LABEL: rocdl.mfma.scale.f32.16x16x128.f8f6f4
  // fp8 * fp8
  // CHECK: call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <4 x float> %{{.*}}, i32 0, i32 0, i32 0, i32 %{{.*}}, i32 0, i32 %{{.*}})
  %r00 = rocdl.mfma.scale.f32.16x16x128.f8f6f4 %arg2, %arg2, %arg1, 0, 0, 0, %arg0, 0, %arg0 :
                              (vector<8xi32>, vector<8xi32>, vector<4xf32>, i32, i32) -> vector<4xf32>

  // fp8 * bf8
  // CHECK: call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <4 x float> %{{.*}}, i32 0, i32 1, i32 0, i32 %{{.*}}, i32 0, i32 %{{.*}})
  %r01 = rocdl.mfma.scale.f32.16x16x128.f8f6f4 %arg2, %arg2, %arg1, 0, 1, 0, %arg0, 0, %arg0 :
                              (vector<8xi32>, vector<8xi32>, vector<4xf32>, i32, i32) -> vector<4xf32>

  // fp8 * fp6
  // CHECK: call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v6i32(<8 x i32> %{{.*}}, <6 x i32> %{{.*}}, <4 x float> %{{.*}}, i32 0, i32 2, i32 0, i32 %{{.*}}, i32 0, i32 %{{.*}})
  %r02 = rocdl.mfma.scale.f32.16x16x128.f8f6f4 %arg2, %arg3, %arg1, 0, 2, 0, %arg0, 0, %arg0 :
                              (vector<8xi32>, vector<6xi32>, vector<4xf32>, i32, i32) -> vector<4xf32>

  // fp8 * bf6
  // CHECK: call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v6i32(<8 x i32> %{{.*}}, <6 x i32> %{{.*}}, <4 x float> %{{.*}}, i32 0, i32 3, i32 0, i32 %{{.*}}, i32 0, i32 %{{.*}})
  %r03 = rocdl.mfma.scale.f32.16x16x128.f8f6f4 %arg2, %arg3, %arg1, 0, 3, 0, %arg0, 0, %arg0 :
                              (vector<8xi32>, vector<6xi32>, vector<4xf32>, i32, i32) -> vector<4xf32>

  // fp8 * fp4
  // CHECK: call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v4i32(<8 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x float> %{{.*}}, i32 0, i32 4, i32 0, i32 %{{.*}}, i32 0, i32 %{{.*}})
  %r04 = rocdl.mfma.scale.f32.16x16x128.f8f6f4 %arg2, %arg4, %arg1, 0, 4, 0, %arg0, 0, %arg0 :
                              (vector<8xi32>, vector<4xi32>, vector<4xf32>, i32, i32) -> vector<4xf32>

  // bf8 * fp8
  // CHECK: call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <4 x float> %{{.*}}, i32 1, i32 0, i32 0, i32 %{{.*}}, i32 0, i32 %{{.*}})
  %r10 = rocdl.mfma.scale.f32.16x16x128.f8f6f4 %arg2, %arg2, %arg1, 1, 0, 0, %arg0, 0, %arg0 :
                              (vector<8xi32>, vector<8xi32>, vector<4xf32>, i32, i32) -> vector<4xf32>

  // bf8 * bf8
  // CHECK: call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <4 x float> %{{.*}}, i32 1, i32 1, i32 0, i32 %{{.*}}, i32 0, i32 %{{.*}})
  %r11 = rocdl.mfma.scale.f32.16x16x128.f8f6f4 %arg2, %arg2, %arg1, 1, 1, 0, %arg0, 0, %arg0 :
                              (vector<8xi32>, vector<8xi32>, vector<4xf32>, i32, i32) -> vector<4xf32>

  // bf8 * fp6
  // CHECK: call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v6i32(<8 x i32> %{{.*}}, <6 x i32> %{{.*}}, <4 x float> %{{.*}}, i32 1, i32 2, i32 0, i32 %{{.*}}, i32 0, i32 %{{.*}})
  %r12 = rocdl.mfma.scale.f32.16x16x128.f8f6f4 %arg2, %arg3, %arg1, 1, 2, 0, %arg0, 0, %arg0 :
                              (vector<8xi32>, vector<6xi32>, vector<4xf32>, i32, i32) -> vector<4xf32>

  // bf8 * bf6
  // CHECK: call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v6i32(<8 x i32> %{{.*}}, <6 x i32> %{{.*}}, <4 x float> %{{.*}}, i32 1, i32 3, i32 0, i32 %{{.*}}, i32 0, i32 %{{.*}})
  %r13 = rocdl.mfma.scale.f32.16x16x128.f8f6f4 %arg2, %arg3, %arg1, 1, 3, 0, %arg0, 0, %arg0 :
                              (vector<8xi32>, vector<6xi32>, vector<4xf32>, i32, i32) -> vector<4xf32>

  // bf8 * fp4
  // CHECK: call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v4i32(<8 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x float> %{{.*}}, i32 1, i32 4, i32 0, i32 %{{.*}}, i32 0, i32 %{{.*}})
  %r14 = rocdl.mfma.scale.f32.16x16x128.f8f6f4 %arg2, %arg4, %arg1, 1, 4, 0, %arg0, 0, %arg0 :
                              (vector<8xi32>, vector<4xi32>, vector<4xf32>, i32, i32) -> vector<4xf32>

  // fp6 * fp8
  // CHECK: call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v6i32.v8i32(<6 x i32> %{{.*}}, <8 x i32> %{{.*}}, <4 x float> %{{.*}}, i32 2, i32 0, i32 0, i32 %{{.*}}, i32 0, i32 %{{.*}})
  %r20 = rocdl.mfma.scale.f32.16x16x128.f8f6f4 %arg3, %arg2, %arg1, 2, 0, 0, %arg0, 0, %arg0 :
                              (vector<6xi32>, vector<8xi32>, vector<4xf32>, i32, i32) -> vector<4xf32>

  // fp6 * bf8
  // CHECK: call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v6i32.v8i32(<6 x i32> %{{.*}}, <8 x i32> %{{.*}}, <4 x float> %{{.*}}, i32 2, i32 1, i32 0, i32 %{{.*}}, i32 0, i32 %{{.*}})
  %r21 = rocdl.mfma.scale.f32.16x16x128.f8f6f4 %arg3, %arg2, %arg1, 2, 1, 0, %arg0, 0, %arg0 :
                              (vector<6xi32>, vector<8xi32>, vector<4xf32>, i32, i32) -> vector<4xf32>

  // fp6 * fp6
  // CHECK: call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v6i32.v6i32(<6 x i32> %{{.*}}, <6 x i32> %{{.*}}, <4 x float> %{{.*}}, i32 2, i32 2, i32 0, i32 %{{.*}}, i32 0, i32 %{{.*}})
  %r22 = rocdl.mfma.scale.f32.16x16x128.f8f6f4 %arg3, %arg3, %arg1, 2, 2, 0, %arg0, 0, %arg0 :
                              (vector<6xi32>, vector<6xi32>, vector<4xf32>, i32, i32) -> vector<4xf32>

  // fp6 * bf6
  // CHECK: call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v6i32.v6i32(<6 x i32> %{{.*}}, <6 x i32> %{{.*}}, <4 x float> %{{.*}}, i32 2, i32 3, i32 0, i32 %{{.*}}, i32 0, i32 %{{.*}})
  %r23 = rocdl.mfma.scale.f32.16x16x128.f8f6f4 %arg3, %arg3, %arg1, 2, 3, 0, %arg0, 0, %arg0 :
                              (vector<6xi32>, vector<6xi32>, vector<4xf32>, i32, i32) -> vector<4xf32>

  // fp6 * fp4
  // CHECK: call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v6i32.v4i32(<6 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x float> %{{.*}}, i32 2, i32 4, i32 0, i32 %{{.*}}, i32 0, i32 %{{.*}})
  %r24 = rocdl.mfma.scale.f32.16x16x128.f8f6f4 %arg3, %arg4, %arg1, 2, 4, 0, %arg0, 0, %arg0 :
                              (vector<6xi32>, vector<4xi32>, vector<4xf32>, i32, i32) -> vector<4xf32>

  // bf6 * fp8
  // CHECK: call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v6i32.v8i32(<6 x i32> %{{.*}}, <8 x i32> %{{.*}}, <4 x float> %{{.*}}, i32 3, i32 0, i32 0, i32 %{{.*}}, i32 0, i32 %{{.*}})
  %r30 = rocdl.mfma.scale.f32.16x16x128.f8f6f4 %arg3, %arg2, %arg1, 3, 0, 0, %arg0, 0, %arg0 :
                              (vector<6xi32>, vector<8xi32>, vector<4xf32>, i32, i32) -> vector<4xf32>

  // bf6 * bf8
  // CHECK: call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v6i32.v8i32(<6 x i32> %{{.*}}, <8 x i32> %{{.*}}, <4 x float> %{{.*}}, i32 3, i32 1, i32 0, i32 %{{.*}}, i32 0, i32 %{{.*}})
  %r31 = rocdl.mfma.scale.f32.16x16x128.f8f6f4 %arg3, %arg2, %arg1, 3, 1, 0, %arg0, 0, %arg0 :
                              (vector<6xi32>, vector<8xi32>, vector<4xf32>, i32, i32) -> vector<4xf32>

  // bf6 * fp6
  // CHECK: call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v6i32.v6i32(<6 x i32> %{{.*}}, <6 x i32> %{{.*}}, <4 x float> %{{.*}}, i32 3, i32 2, i32 0, i32 %{{.*}}, i32 0, i32 %{{.*}})
  %r32 = rocdl.mfma.scale.f32.16x16x128.f8f6f4 %arg3, %arg3, %arg1, 3, 2, 0, %arg0, 0, %arg0 :
                              (vector<6xi32>, vector<6xi32>, vector<4xf32>, i32, i32) -> vector<4xf32>

  // bf6 * bf6
  // CHECK: call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v6i32.v6i32(<6 x i32> %{{.*}}, <6 x i32> %{{.*}}, <4 x float> %{{.*}}, i32 3, i32 3, i32 0, i32 %{{.*}}, i32 0, i32 %{{.*}})
  %r33 = rocdl.mfma.scale.f32.16x16x128.f8f6f4 %arg3, %arg3, %arg1, 3, 3, 0, %arg0, 0, %arg0 :
                              (vector<6xi32>, vector<6xi32>, vector<4xf32>, i32, i32) -> vector<4xf32>

  // bf6 * fp4
  // CHECK: call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v6i32.v4i32(<6 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x float> %{{.*}}, i32 3, i32 4, i32 0, i32 %{{.*}}, i32 0, i32 %{{.*}})
  %r34 = rocdl.mfma.scale.f32.16x16x128.f8f6f4 %arg3, %arg4, %arg1, 3, 4, 0, %arg0, 0, %arg0 :
                              (vector<6xi32>, vector<4xi32>, vector<4xf32>, i32, i32) -> vector<4xf32>

  // fp4 * fp8
  // CHECK: call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v8i32(<4 x i32> %{{.*}}, <8 x i32> %{{.*}}, <4 x float> %{{.*}}, i32 4, i32 0, i32 0, i32 %{{.*}}, i32 0, i32 %{{.*}})
  %r40 = rocdl.mfma.scale.f32.16x16x128.f8f6f4 %arg4, %arg2, %arg1, 4, 0, 0, %arg0, 0, %arg0 :
                              (vector<4xi32>, vector<8xi32>, vector<4xf32>, i32, i32) -> vector<4xf32>

  // fp4 * bf8
  // CHECK: call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v8i32(<4 x i32> %{{.*}}, <8 x i32> %{{.*}}, <4 x float> %{{.*}}, i32 4, i32 1, i32 0, i32 %{{.*}}, i32 0, i32 %{{.*}})
  %r41 = rocdl.mfma.scale.f32.16x16x128.f8f6f4 %arg4, %arg2, %arg1, 4, 1, 0, %arg0, 0, %arg0 :
                              (vector<4xi32>, vector<8xi32>, vector<4xf32>, i32, i32) -> vector<4xf32>

  // fp4 * fp6
  // CHECK: call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v6i32(<4 x i32> %{{.*}}, <6 x i32> %{{.*}}, <4 x float> %{{.*}}, i32 4, i32 2, i32 0, i32 %{{.*}}, i32 0, i32 %{{.*}})
  %r42 = rocdl.mfma.scale.f32.16x16x128.f8f6f4 %arg4, %arg3, %arg1, 4, 2, 0, %arg0, 0, %arg0 :
                              (vector<4xi32>, vector<6xi32>, vector<4xf32>, i32, i32) -> vector<4xf32>

  // fp4 * bf6
  // CHECK: call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v6i32(<4 x i32> %{{.*}}, <6 x i32> %{{.*}}, <4 x float> %{{.*}}, i32 4, i32 3, i32 0, i32 %{{.*}}, i32 0, i32 %{{.*}})
  %r43 = rocdl.mfma.scale.f32.16x16x128.f8f6f4 %arg4, %arg3, %arg1, 4, 3, 0, %arg0, 0, %arg0 :
                              (vector<4xi32>, vector<6xi32>, vector<4xf32>, i32, i32) -> vector<4xf32>

  // fp4 * fp4
  // CHECK: call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x float> %{{.*}}, i32 4, i32 4, i32 0, i32 %{{.*}}, i32 0, i32 %{{.*}}
  %r44 = rocdl.mfma.scale.f32.16x16x128.f8f6f4 %arg4, %arg4, %arg1, 4, 4, 0, %arg0, 0, %arg0 :
                              (vector<4xi32>, vector<4xi32>, vector<4xf32>, i32, i32) -> vector<4xf32>

  llvm.return %r00 : vector<4 x f32>
}

llvm.func @rocdl.wmma(%arg0 : vector<8xf32>, %arg1 : vector<16 x f16>, %arg2 : vector<16 x i16>, %arg3 : vector<8 x i32>,
                      %arg4 : vector<2xi32>, %arg5 : vector<4xi32>, %arg6 : vector<4xf32>, %arg7 : vector<8xf16>, %arg8 : vector<8xi16>,
                      %arg9 : vector<32xf16>, %arg10 : vector<16xf32>, %arg11 : vector<4xf32>, %arg12 : vector<32xf32>, %arg13 : vector<64xf32>,
                      %arg14 : vector<64xi32>, %arg15 : vector<64xf16>, %arg16 : vector<16xbf16>, %arg17 : vector<32xbf16>) -> vector<8xf32> {

  // ---- Wave32 -----
  // f16 -> f32
  // CHECK: call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v16f16(<16 x half> %{{.*}} <16 x half> %{{.*}} <8 x float> %{{.*}})
  %r0 = rocdl.wmma.f32.16x16x16.f16 %arg1, %arg1, %arg0 : (vector<16xf16>, vector<16xf16>, vector<8xf32>) -> vector<8xf32>

  // bf16 -> f32
  // CHECK: call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32.v16i16(<16 x i16> %{{.*}} <16 x i16> %{{.*}} <8 x float> %{{.*}})
  %r1 = rocdl.wmma.f32.16x16x16.bf16 %arg2, %arg2, %arg0 : (vector<16xi16>, vector<16xi16>, vector<8xf32>) -> vector<8xf32>

  // f16 -> f16 (OPSEL = {0,1})
  // CHECK: call <16 x half> @llvm.amdgcn.wmma.f16.16x16x16.f16.v16f16.v16f16(<16 x half> %{{.*}} <16 x half> %{{.*}} <16 x half> %{{.*}} i1 false)
  %r2 = rocdl.wmma.f16.16x16x16.f16 %arg1, %arg1, %arg1 {opsel = false} : (vector<16xf16>, vector<16xf16>, vector<16xf16>) -> vector<16xf16>

  // bf16 -> bf16 (OPSEL = {0,1})
  // CHECK: call <16 x i16> @llvm.amdgcn.wmma.bf16.16x16x16.bf16.v16i16.v16i16(<16 x i16> %{{.*}} <16 x i16> %{{.*}} <16 x i16> %{{.*}} i1 false)
  %r4 = rocdl.wmma.bf16.16x16x16.bf16 %arg2, %arg2, %arg2 {opsel = false} : (vector<16xi16>, vector<16xi16>, vector<16xi16>) -> vector<16xi16>

  // int8 -> int32 (signA = {0,1}, signB = {0,1}, clamp = {0,1})
  // CHECK: call <8 x i32> @llvm.amdgcn.wmma.i32.16x16x16.iu8.v8i32.v4i32(i1 false, <4 x i32> %{{.*}} i1 false, <4 x i32> %{{.*}} <8 x i32> %{{.*}} i1 false)
  %r5 = rocdl.wmma.i32.16x16x16.iu8 %arg5, %arg5, %arg3 {signA = false, signB = false, clamp = false} : (vector<4xi32>, vector<4xi32>, vector<8xi32>) -> vector<8xi32>

  // int4 -> int32 (signA = {0,1}, signB = {0,1}, clamp = {0,1})
  // CHECK: call <8 x i32> @llvm.amdgcn.wmma.i32.16x16x16.iu4.v8i32.v2i32(i1 false, <2 x i32> %{{.*}} i1 false, <2 x i32> %{{.*}} <8 x i32> %{{.*}} i1 false)
  %r6 = rocdl.wmma.i32.16x16x16.iu4 %arg4, %arg4, %arg3 {signA = false, signB = false, clamp = false} : (vector<2xi32>, vector<2xi32>, vector<8xi32>) -> vector<8xi32>

  // int4 -> int32 (signA = {0,1}, signB = {0,1}, clamp = {0,1})
  // CHECK: call <8 x i32> @llvm.amdgcn.wmma.i32.16x16x32.iu4.v8i32.v2i32(i1 false, <2 x i32> %{{.*}} i1 false, <2 x i32> %{{.*}} <8 x i32> %{{.*}} i1 false)
  %r6.gfx12 = rocdl.wmma.i32.16x16x32.iu4 %arg4, %arg4, %arg3 {signA = false, signB = false, clamp = false} : (vector<2xi32>, vector<2xi32>, vector<8xi32>) -> vector<8xi32>

  // Test signA=true, signB=false for iu8
  // CHECK: call <8 x i32> @llvm.amdgcn.wmma.i32.16x16x16.iu8.v8i32.v4i32(i1 true, <4 x i32> %{{.*}} i1 false, <4 x i32> %{{.*}} <8 x i32> %{{.*}} i1 false)
  %r5a = rocdl.wmma.i32.16x16x16.iu8 %arg5, %arg5, %arg3 {signA = true, signB = false, clamp = false} : (vector<4xi32>, vector<4xi32>, vector<8xi32>) -> vector<8xi32>

  // Test signA=false, signB=true for iu8
  // CHECK: call <8 x i32> @llvm.amdgcn.wmma.i32.16x16x16.iu8.v8i32.v4i32(i1 false, <4 x i32> %{{.*}} i1 true, <4 x i32> %{{.*}} <8 x i32> %{{.*}} i1 false)
  %r5b = rocdl.wmma.i32.16x16x16.iu8 %arg5, %arg5, %arg3 {signA = false, signB = true, clamp = false} : (vector<4xi32>, vector<4xi32>, vector<8xi32>) -> vector<8xi32>

  // Test signA=true, signB=true, clamp=true for iu8
  // CHECK: call <8 x i32> @llvm.amdgcn.wmma.i32.16x16x16.iu8.v8i32.v4i32(i1 true, <4 x i32> %{{.*}} i1 true, <4 x i32> %{{.*}} <8 x i32> %{{.*}} i1 true)
  %r5c = rocdl.wmma.i32.16x16x16.iu8 %arg5, %arg5, %arg3 {signA = true, signB = true, clamp = true} : (vector<4xi32>, vector<4xi32>, vector<8xi32>) -> vector<8xi32>

  // Test signA=true, signB=false for iu4
  // CHECK: call <8 x i32> @llvm.amdgcn.wmma.i32.16x16x16.iu4.v8i32.v2i32(i1 true, <2 x i32> %{{.*}} i1 false, <2 x i32> %{{.*}} <8 x i32> %{{.*}} i1 false)
  %r6a = rocdl.wmma.i32.16x16x16.iu4 %arg4, %arg4, %arg3 {signA = true, signB = false, clamp = false} : (vector<2xi32>, vector<2xi32>, vector<8xi32>) -> vector<8xi32>

  // Test signA=false, signB=true, clamp=true for iu4
  // CHECK: call <8 x i32> @llvm.amdgcn.wmma.i32.16x16x16.iu4.v8i32.v2i32(i1 false, <2 x i32> %{{.*}} i1 true, <2 x i32> %{{.*}} <8 x i32> %{{.*}} i1 true)
  %r6b = rocdl.wmma.i32.16x16x16.iu4 %arg4, %arg4, %arg3 {signA = false, signB = true, clamp = true} : (vector<2xi32>, vector<2xi32>, vector<8xi32>) -> vector<8xi32>

  // Test signA=true, signB=true for iu4 gfx12
  // CHECK: call <8 x i32> @llvm.amdgcn.wmma.i32.16x16x32.iu4.v8i32.v2i32(i1 true, <2 x i32> %{{.*}} i1 true, <2 x i32> %{{.*}} <8 x i32> %{{.*}} i1 false)
  %r6c = rocdl.wmma.i32.16x16x32.iu4 %arg4, %arg4, %arg3 {signA = true, signB = true, clamp = false} : (vector<2xi32>, vector<2xi32>, vector<8xi32>) -> vector<8xi32>

  // f32 -> f32
  // CHECK: call <4 x float> @llvm.amdgcn.wmma.f32.16x16x4.f32.v4f32.v16f32(i1 false, <16 x float> %{{.*}} i1 false, <16 x float> %{{.*}} i16 0, <4 x float> %{{.*}} i1 false, i1 false)
  %r1.gfx1250 = rocdl.wmma.f32.16x16x4.f32 %arg10, %arg10, %arg11 {signA = false, signB = false, modC = 0 : i16} : (vector<16xf32>, vector<16xf32>, vector<4xf32>) -> vector<4xf32>

  // f16 -> f32
  // CHECK: call <32 x float> @llvm.amdgcn.wmma.f32.16x16x32.f16.v32f32.v16f16(i1 false, <16 x half> %{{.*}} i1 false, <16 x half> %{{.*}} i16 0, <32 x float> %{{.*}} i1 false, i1 false)
  %r2.gfx1250 = rocdl.wmma.f32.16x16x32.f16 %arg1, %arg1, %arg12 {signA = false, signB = false, modC = 0 : i16} : (vector<16xf16>, vector<16xf16>, vector<32xf32>) -> vector<32xf32>

  // bf16 -> f32
  // CHECK: call <32 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v32f32.v16bf16(i1 false, <16 x bfloat> %{{.*}} i1 false, <16 x bfloat> %{{.*}} i16 0, <32 x float> %{{.*}} i1 false, i1 false)
  %r3.gfx1250 = rocdl.wmma.f32.16x16x32.bf16 %arg16, %arg16, %arg12 {signA = false, signB = false, modC = 0 : i16} : (vector<16xbf16>, vector<16xbf16>, vector<32xf32>) -> vector<32xf32>

  // f16 -> f16
  // CHECK: call <32 x half> @llvm.amdgcn.wmma.f16.16x16x32.f16.v32f16.v16f16(i1 false, <16 x half> %{{.*}} i1 false, <16 x half> %{{.*}} i16 0, <32 x half> %{{.*}} i1 false, i1 false)
  %r4.gfx1250 = rocdl.wmma.f16.16x16x32.f16 %arg1, %arg1, %arg9 {signA = false, signB = false, modC = 0 : i16} : (vector<16xf16>, vector<16xf16>, vector<32xf16>) -> vector<32xf16>

  // bf16 -> bf16
  // CHECK: call <32 x bfloat> @llvm.amdgcn.wmma.bf16.16x16x32.bf16.v32bf16.v16bf16(i1 false, <16 x bfloat> %{{.*}} i1 false, <16 x bfloat> %{{.*}} i16 0, <32 x bfloat> %{{.*}} i1 false, i1 false)
  %r5.gfx1250 = rocdl.wmma.bf16.16x16x32.bf16 %arg16, %arg16, %arg17 {signA = false, signB = false, modC = 0 : i16} : (vector<16xbf16>, vector<16xbf16>, vector<32xbf16>) -> vector<32xbf16>

  // bf16 -> bf16 / f32
  // CHECK: call <32 x bfloat> @llvm.amdgcn.wmma.bf16f32.16x16x32.bf16.v32bf16.v16bf16.v32f32(i1 false, <16 x bfloat> %{{.*}} i1 false, <16 x bfloat> %{{.*}} i16 0, <32 x float> %{{.*}} i1 false, i1 false)
  %r6.gfx1250 = rocdl.wmma.bf16f32.16x16x32.bf16 %arg16, %arg16, %arg12 {signA = false, signB = false, modC = 0 : i16} : (vector<16xbf16>, vector<16xbf16>, vector<32xf32>) -> vector<32xbf16>

  // f8/bf8 -> f16/f32
  // CHECK: call <64 x float> @llvm.amdgcn.wmma.f32.16x16x64.fp8.fp8.v64f32.v4i32(<4 x i32> %{{.*}} <4 x i32> %{{.*}} i16 0, <64 x float> %{{.*}} i1 false, i1 false)
  %r7.gfx1250 = rocdl.wmma.f32.16x16x64.fp8_fp8 %arg5, %arg5, %arg13 {signA = false, signB = false, modC = 0 : i16} : (vector<4xi32>, vector<4xi32>, vector<64xf32>) -> vector<64xf32>

  // CHECK: call <64 x float> @llvm.amdgcn.wmma.f32.16x16x64.fp8.bf8.v64f32.v4i32(<4 x i32> %{{.*}} <4 x i32> %{{.*}} i16 0, <64 x float> %{{.*}} i1 false, i1 false)
  %r8.gfx1250 = rocdl.wmma.f32.16x16x64.fp8_bf8 %arg5, %arg5, %arg13 {signA = false, signB = false, modC = 0 : i16} : (vector<4xi32>, vector<4xi32>, vector<64xf32>) -> vector<64xf32>

  // CHECK: call <64 x float> @llvm.amdgcn.wmma.f32.16x16x64.bf8.fp8.v64f32.v4i32(<4 x i32> %{{.*}} <4 x i32> %{{.*}} i16 0, <64 x float> %{{.*}} i1 false, i1 false)
  %r9.gfx1250 = rocdl.wmma.f32.16x16x64.bf8_fp8 %arg5, %arg5, %arg13 {signA = false, signB = false, modC = 0 : i16} : (vector<4xi32>, vector<4xi32>, vector<64xf32>) -> vector<64xf32>

  // CHECK: call <64 x float> @llvm.amdgcn.wmma.f32.16x16x64.bf8.bf8.v64f32.v4i32(<4 x i32> %{{.*}} <4 x i32> %{{.*}} i16 0, <64 x float> %{{.*}} i1 false, i1 false)
  %r10.gfx1250 = rocdl.wmma.f32.16x16x64.bf8_bf8 %arg5, %arg5, %arg13 {signA = false, signB = false, modC = 0 : i16} : (vector<4xi32>, vector<4xi32>, vector<64xf32>) -> vector<64xf32>

  // CHECK: call <64 x half> @llvm.amdgcn.wmma.f16.16x16x64.fp8.fp8.v64f16.v4i32(<4 x i32> %{{.*}} <4 x i32> %{{.*}} i16 0, <64 x half> %{{.*}} i1 false, i1 false)
  %r11.gfx1250 = rocdl.wmma.f16.16x16x64.fp8_fp8 %arg5, %arg5, %arg15 {signA = false, signB = false, modC = 0 : i16} : (vector<4xi32>, vector<4xi32>, vector<64xf16>) -> vector<64xf16>

  // CHECK: call <64 x half> @llvm.amdgcn.wmma.f16.16x16x64.fp8.bf8.v64f16.v4i32(<4 x i32> %{{.*}} <4 x i32> %{{.*}} i16 0, <64 x half> %{{.*}} i1 false, i1 false)
  %r12.gfx1250 = rocdl.wmma.f16.16x16x64.fp8_bf8 %arg5, %arg5, %arg15 {signA = false, signB = false, modC = 0 : i16} : (vector<4xi32>, vector<4xi32>, vector<64xf16>) -> vector<64xf16>

  // CHECK: call <64 x half> @llvm.amdgcn.wmma.f16.16x16x64.bf8.fp8.v64f16.v4i32(<4 x i32> %{{.*}} <4 x i32> %{{.*}} i16 0, <64 x half> %{{.*}} i1 false, i1 false)
  %r13.gfx1250 = rocdl.wmma.f16.16x16x64.bf8_fp8 %arg5, %arg5, %arg15 {signA = false, signB = false, modC = 0 : i16} : (vector<4xi32>, vector<4xi32>, vector<64xf16>) -> vector<64xf16>

  // CHECK: call <64 x half> @llvm.amdgcn.wmma.f16.16x16x64.bf8.bf8.v64f16.v4i32(<4 x i32> %{{.*}} <4 x i32> %{{.*}} i16 0, <64 x half> %{{.*}} i1 false, i1 false)
  %r14.gfx1250 = rocdl.wmma.f16.16x16x64.bf8_bf8 %arg5, %arg5, %arg15 {signA = false, signB = false, modC = 0 : i16} : (vector<4xi32>, vector<4xi32>, vector<64xf16>) -> vector<64xf16>

  // CHECK: call <64 x float> @llvm.amdgcn.wmma.f32.16x16x128.fp8.fp8.v64f32.v4i32(<4 x i32> %{{.*}} <4 x i32> %{{.*}} i16 0, <64 x float> %{{.*}} i1 false, i1 false)
  %r15.gfx1250 = rocdl.wmma.f32.16x16x128.fp8_fp8 %arg5, %arg5, %arg13 {signA = false, signB = false, modC = 0 : i16} : (vector<4xi32>, vector<4xi32>, vector<64xf32>) -> vector<64xf32>

  // CHECK: call <64 x float> @llvm.amdgcn.wmma.f32.16x16x128.fp8.bf8.v64f32.v4i32(<4 x i32> %{{.*}} <4 x i32> %{{.*}} i16 0, <64 x float> %{{.*}} i1 false, i1 false)
  %r16.gfx1250 = rocdl.wmma.f32.16x16x128.fp8_bf8 %arg5, %arg5, %arg13 {signA = false, signB = false, modC = 0 : i16} : (vector<4xi32>, vector<4xi32>, vector<64xf32>) -> vector<64xf32>

  // CHECK: call <64 x float> @llvm.amdgcn.wmma.f32.16x16x128.bf8.fp8.v64f32.v4i32(<4 x i32> %{{.*}} <4 x i32> %{{.*}} i16 0, <64 x float> %{{.*}} i1 false, i1 false)
  %r17.gfx1250 = rocdl.wmma.f32.16x16x128.bf8_fp8 %arg5, %arg5, %arg13 {signA = false, signB = false, modC = 0 : i16} : (vector<4xi32>, vector<4xi32>, vector<64xf32>) -> vector<64xf32>

  // CHECK: call <64 x float> @llvm.amdgcn.wmma.f32.16x16x128.bf8.bf8.v64f32.v4i32(<4 x i32> %{{.*}} <4 x i32> %{{.*}} i16 0, <64 x float> %{{.*}} i1 false, i1 false)
  %r18.gfx1250 = rocdl.wmma.f32.16x16x128.bf8_bf8 %arg5, %arg5, %arg13 {signA = false, signB = false, modC = 0 : i16} : (vector<4xi32>, vector<4xi32>, vector<64xf32>) -> vector<64xf32>

  // CHECK: call <64 x half> @llvm.amdgcn.wmma.f16.16x16x128.fp8.fp8.v64f16.v4i32(<4 x i32> %{{.*}} <4 x i32> %{{.*}} i16 0, <64 x half> %{{.*}} i1 false, i1 false)
  %r19.gfx1250 = rocdl.wmma.f16.16x16x128.fp8_fp8 %arg5, %arg5, %arg15 {signA = false, signB = false, modC = 0 : i16} : (vector<4xi32>, vector<4xi32>, vector<64xf16>) -> vector<64xf16>

  // CHECK: call <64 x half> @llvm.amdgcn.wmma.f16.16x16x128.fp8.bf8.v64f16.v4i32(<4 x i32> %{{.*}} <4 x i32> %{{.*}} i16 0, <64 x half> %{{.*}} i1 false, i1 false)
  %r20.gfx1250 = rocdl.wmma.f16.16x16x128.fp8_bf8 %arg5, %arg5, %arg15 {signA = false, signB = false, modC = 0 : i16} : (vector<4xi32>, vector<4xi32>, vector<64xf16>) -> vector<64xf16>

  // CHECK: call <64 x half> @llvm.amdgcn.wmma.f16.16x16x128.bf8.fp8.v64f16.v4i32(<4 x i32> %{{.*}} <4 x i32> %{{.*}} i16 0, <64 x half> %{{.*}} i1 false, i1 false)
  %r21.gfx1250 = rocdl.wmma.f16.16x16x128.bf8_fp8 %arg5, %arg5, %arg15 {signA = false, signB = false, modC = 0 : i16} : (vector<4xi32>, vector<4xi32>, vector<64xf16>) -> vector<64xf16>

  // CHECK: call <64 x half> @llvm.amdgcn.wmma.f16.16x16x128.bf8.bf8.v64f16.v4i32(<4 x i32> %{{.*}} <4 x i32> %{{.*}} i16 0, <64 x half> %{{.*}} i1 false, i1 false)
  %r22.gfx1250 = rocdl.wmma.f16.16x16x128.bf8_bf8 %arg5, %arg5, %arg15 {signA = false, signB = false, modC = 0 : i16} : (vector<4xi32>, vector<4xi32>, vector<64xf16>) -> vector<64xf16>

  // iu8 -> i32
  // CHECK: call <64 x i32> @llvm.amdgcn.wmma.i32.16x16x64.iu8.v64i32.v4i32(i1 false, <4 x i32> %{{.*}} i1 false, <4 x i32> %{{.*}} <64 x i32> %{{.*}} i1 false, i1 false, i1 false)
  %r23.gfx1250 = rocdl.wmma.i32.16x16x64.iu8 %arg5, %arg5, %arg14 {signA = false, signB = false, clamp=false} : (vector<4xi32>, vector<4xi32>, vector<64xi32>) -> vector<64xi32>

  // Test signA=true, signB=true for iu8 gfx1250
  // CHECK: call <64 x i32> @llvm.amdgcn.wmma.i32.16x16x64.iu8.v64i32.v4i32(i1 true, <4 x i32> %{{.*}} i1 true, <4 x i32> %{{.*}} <64 x i32> %{{.*}} i1 false, i1 false, i1 false)
  %r23a.gfx1250 = rocdl.wmma.i32.16x16x64.iu8 %arg5, %arg5, %arg14 {signA = true, signB = true, clamp=false} : (vector<4xi32>, vector<4xi32>, vector<64xi32>) -> vector<64xi32>

  // Test signA=true, signB=false, reuseA=true, reuseB=true for iu8 gfx1250
  // CHECK: call <64 x i32> @llvm.amdgcn.wmma.i32.16x16x64.iu8.v64i32.v4i32(i1 true, <4 x i32> %{{.*}} i1 false, <4 x i32> %{{.*}} <64 x i32> %{{.*}} i1 true, i1 true, i1 false)
  %r23b.gfx1250 = rocdl.wmma.i32.16x16x64.iu8 %arg5, %arg5, %arg14 {signA = true, signB = false, reuseA = true, reuseB = true, clamp=false} : (vector<4xi32>, vector<4xi32>, vector<64xi32>) -> vector<64xi32>

  // Test signA=true, signB=true with modC=1 for f32 gfx1250
  // CHECK: call <4 x float> @llvm.amdgcn.wmma.f32.16x16x4.f32.v4f32.v16f32(i1 true, <16 x float> %{{.*}} i1 true, <16 x float> %{{.*}} i16 1, <4 x float> %{{.*}} i1 false, i1 false)
  %r1a.gfx1250 = rocdl.wmma.f32.16x16x4.f32 %arg10, %arg10, %arg11 {signA = true, signB = true, modC = 1 : i16, reuseA = false, reuseB = false} : (vector<16xf32>, vector<16xf32>, vector<4xf32>) -> vector<4xf32>

  // Test with modC=2 and signA=false, signB=true, reuseA=true for f16 gfx1250
  // CHECK: call <32 x float> @llvm.amdgcn.wmma.f32.16x16x32.f16.v32f32.v16f16(i1 false, <16 x half> %{{.*}} i1 true, <16 x half> %{{.*}} i16 2, <32 x float> %{{.*}} i1 true, i1 false)
  %r2a.gfx1250 = rocdl.wmma.f32.16x16x32.f16 %arg1, %arg1, %arg12 {signA = false, signB = true, modC = 2 : i16, reuseA = true, reuseB = false} : (vector<16xf16>, vector<16xf16>, vector<32xf32>) -> vector<32xf32>

  // Test with modC=3 and signA=true, signB=true, reuseB=true for bf16 gfx1250
  // CHECK: call <32 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v32f32.v16bf16(i1 true, <16 x bfloat> %{{.*}} i1 true, <16 x bfloat> %{{.*}} i16 3, <32 x float> %{{.*}} i1 false, i1 true)
  %r3a.gfx1250 = rocdl.wmma.f32.16x16x32.bf16 %arg16, %arg16, %arg12 {signA = true, signB = true, modC = 3 : i16, reuseA = false, reuseB = true} : (vector<16xbf16>, vector<16xbf16>, vector<32xf32>) -> vector<32xf32>

  // ---- Wave64 -----

  // f16 -> f32
  // CHECK: call <4 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v4f32.v16f16(<16 x half> %{{.*}} <16 x half> %{{.*}} <4 x float> %{{.*}})
  %r7 = rocdl.wmma.f32.16x16x16.f16 %arg1, %arg1, %arg6 : (vector<16xf16>, vector<16xf16>, vector<4xf32>) -> vector<4xf32>

  // bf16 -> f32
  // CHECK: call <4 x float> @llvm.amdgcn.wmma.f32.16x16x16.bf16.v4f32.v16i16(<16 x i16> %{{.*}} <16 x i16> %{{.*}} <4 x float> %{{.*}})
  %r8 = rocdl.wmma.f32.16x16x16.bf16 %arg2, %arg2, %arg6 : (vector<16xi16>, vector<16xi16>, vector<4xf32>) -> vector<4xf32>

  // f16 -> f16 (OPSEL = {0,1})
  // CHECK: call <8 x half> @llvm.amdgcn.wmma.f16.16x16x16.f16.v8f16.v16f16(<16 x half> %{{.*}} <16 x half> %{{.*}} <8 x half> %{{.*}} i1 false)
  %r9 = rocdl.wmma.f16.16x16x16.f16 %arg1, %arg1, %arg7 {opsel = false} : (vector<16xf16>, vector<16xf16>, vector<8xf16>) -> vector<8xf16>

  // bf16 -> bf16 (OPSEL = {0,1})
  // CHECK: call <8 x i16> @llvm.amdgcn.wmma.bf16.16x16x16.bf16.v8i16.v16i16(<16 x i16> %{{.*}} <16 x i16> %{{.*}} <8 x i16> %{{.*}} i1 false)
  %r11 = rocdl.wmma.bf16.16x16x16.bf16 %arg2, %arg2, %arg8 {opsel = false} : (vector<16xi16>, vector<16xi16>, vector<8xi16>) -> vector<8xi16>

  // int8 -> int32 (signA = {0,1}, signB = {0,1}, clamp = {0,1})
  // CHECK: call <4 x i32> @llvm.amdgcn.wmma.i32.16x16x16.iu8.v4i32.v4i32(i1 false, <4 x i32> %{{.*}} i1 false, <4 x i32> %{{.*}} <4 x i32> %{{.*}} i1 true)
  %r12 = rocdl.wmma.i32.16x16x16.iu8 %arg5, %arg5, %arg5 {signA = false, signB = false, clamp = true} : (vector<4xi32>, vector<4xi32>, vector<4xi32>) -> vector<4xi32>

  // int4 -> int32 (signA = {0,1}, signB = {0,1}, clamp = {0,1})
  // CHECK: call <4 x i32> @llvm.amdgcn.wmma.i32.16x16x16.iu4.v4i32.v2i32(i1 false, <2 x i32> %{{.*}} i1 false, <2 x i32> %{{.*}} <4 x i32> %{{.*}} i1 true)
  %r13 = rocdl.wmma.i32.16x16x16.iu4 %arg4, %arg4, %arg5 {signA = false, signB = false, clamp = true} : (vector<2xi32>, vector<2xi32>, vector<4xi32>) -> vector<4xi32>

  llvm.return %r0 : vector<8xf32>
}

llvm.func @rocdl.ds.read.tr(%ptr : !llvm.ptr<3>) -> vector<4xf16> {
  // CHECK-LABEL: rocdl.ds.read.tr
  // CHECK: call <2 x i32> @llvm.amdgcn.ds.read.tr4.b64.v2i32(ptr addrspace(3) %0)
  %r0 = rocdl.ds.read.tr4.b64 %ptr : !llvm.ptr<3> -> vector<2xi32>
  // CHECK: call <3 x i32> @llvm.amdgcn.ds.read.tr6.b96.v3i32(ptr addrspace(3) %0)
  %r1 = rocdl.ds.read.tr6.b96 %ptr : !llvm.ptr<3> -> vector<3xi32>
  // CHECK: call <2 x i32> @llvm.amdgcn.ds.read.tr8.b64.v2i32(ptr addrspace(3) %0)
  %r2 = rocdl.ds.read.tr8.b64 %ptr : !llvm.ptr<3> -> vector<2xi32>
  // CHECK: call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) %0)
  %r3 = rocdl.ds.read.tr16.b64 %ptr : !llvm.ptr<3> -> vector<4xf16>
  // CHECK: call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) %0)
  %r4 = rocdl.ds.read.tr16.b64 %ptr : !llvm.ptr<3> -> vector<4xbf16>
  llvm.return %r3 : vector<4xf16>
}

llvm.func @rocdl.load.tr.ops(%gl_ptr : !llvm.ptr<1>, %ds_ptr : !llvm.ptr<3>) {
  // CHECK-LABEL: rocdl.load.tr.ops
  // CHECK-SAME: (ptr addrspace(1) %[[GL_PTR:.+]], ptr addrspace(3) %[[DS_PTR:.+]])
  // CHECK: call <2 x i32> @llvm.amdgcn.global.load.tr4.b64.v2i32(ptr addrspace(1) %[[GL_PTR]])
  // CHECK: call <2 x i32> @llvm.amdgcn.global.load.tr.b64.v2i32(ptr addrspace(1) %[[GL_PTR]])
  // CHECK: call <3 x i32> @llvm.amdgcn.global.load.tr6.b96.v3i32(ptr addrspace(1) %[[GL_PTR]])
  // CHECK: call <8 x i16> @llvm.amdgcn.global.load.tr.b128.v8i16(ptr addrspace(1) %[[GL_PTR]])
  // CHECK: call <8 x half> @llvm.amdgcn.global.load.tr.b128.v8f16(ptr addrspace(1) %[[GL_PTR]])
  // CHECK: call <8 x bfloat> @llvm.amdgcn.global.load.tr.b128.v8bf16(ptr addrspace(1) %[[GL_PTR]])

  // CHECK: call <2 x i32> @llvm.amdgcn.ds.load.tr4.b64.v2i32(ptr addrspace(3) %[[DS_PTR]])
  // CHECK: call <2 x i32> @llvm.amdgcn.ds.load.tr8.b64.v2i32(ptr addrspace(3) %[[DS_PTR]])
  // CHECK: call <3 x i32> @llvm.amdgcn.ds.load.tr6.b96.v3i32(ptr addrspace(3) %[[DS_PTR]])
  // CHECK: call <8 x i16> @llvm.amdgcn.ds.load.tr16.b128.v8i16(ptr addrspace(3) %[[DS_PTR]])
  // CHECK: call <8 x half> @llvm.amdgcn.ds.load.tr16.b128.v8f16(ptr addrspace(3) %[[DS_PTR]])
  // CHECK: call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %[[DS_PTR]])

  rocdl.global.load.tr4.b64 %gl_ptr : !llvm.ptr<1> -> vector<2xi32>
  rocdl.global.load.tr.b64 %gl_ptr : !llvm.ptr<1> -> vector<2xi32>
  rocdl.global.load.tr6.b96 %gl_ptr : !llvm.ptr<1> -> vector<3xi32>
  rocdl.global.load.tr.b128 %gl_ptr : !llvm.ptr<1> -> vector<8xi16>
  rocdl.global.load.tr.b128 %gl_ptr : !llvm.ptr<1> -> vector<8xf16>
  rocdl.global.load.tr.b128 %gl_ptr : !llvm.ptr<1> -> vector<8xbf16>

  rocdl.ds.load.tr4.b64 %ds_ptr : !llvm.ptr<3> -> vector<2xi32>
  rocdl.ds.load.tr8.b64 %ds_ptr : !llvm.ptr<3> -> vector<2xi32>
  rocdl.ds.load.tr6.b96 %ds_ptr : !llvm.ptr<3> -> vector<3xi32>
  rocdl.ds.load.tr16.b128 %ds_ptr : !llvm.ptr<3> -> vector<8xi16>
  rocdl.ds.load.tr16.b128 %ds_ptr : !llvm.ptr<3> -> vector<8xf16>
  rocdl.ds.load.tr16.b128 %ds_ptr : !llvm.ptr<3> -> vector<8xbf16>
  llvm.return
}

llvm.func @rocdl.load.to.lds(%src : !llvm.ptr<7>, %dst: !llvm.ptr<3>) {
  //CHECK: call void @llvm.amdgcn.load.to.lds.p7
  rocdl.load.to.lds %src, %dst, 4, 0, 0 : !llvm.ptr<7>
  llvm.return
}

llvm.func @rocdl.global.load.lds(%src : !llvm.ptr<1>, %dst: !llvm.ptr<3>) {
  //CHECK: call void @llvm.amdgcn.global.load.lds
  rocdl.global.load.lds %src, %dst, 4, 0, 0
  llvm.return
}

// CHECK-LABEL: rocdl.global.load.async.to.lds
llvm.func @rocdl.global.load.async.to.lds(%src : !llvm.ptr<1>, %dst: !llvm.ptr<3>) {
  // CHECK: call void @llvm.amdgcn.global.load.async.to.lds.b8
  rocdl.global.load.async.to.lds.b8 %src, %dst, 0, 0 : !llvm.ptr<1>, !llvm.ptr<3>
  // CHECK: call void @llvm.amdgcn.global.load.async.to.lds.b32
  rocdl.global.load.async.to.lds.b32 %src, %dst, 0, 0 : !llvm.ptr<1>, !llvm.ptr<3>
  // CHECK: call void @llvm.amdgcn.global.load.async.to.lds.b64
  rocdl.global.load.async.to.lds.b64 %src, %dst, 0, 0 : !llvm.ptr<1>, !llvm.ptr<3>
  // CHECK: call void @llvm.amdgcn.global.load.async.to.lds.b128
  rocdl.global.load.async.to.lds.b128 %src, %dst, 0, 0 : !llvm.ptr<1>, !llvm.ptr<3>
  llvm.return
}

// CHECK-LABEL: rocdl.cluster.load.async.to.lds
llvm.func @rocdl.cluster.load.async.to.lds(%src : !llvm.ptr<1>, %dst: !llvm.ptr<3>, %mask: i32) {
  // CHECK: call void @llvm.amdgcn.cluster.load.async.to.lds.b8
  rocdl.cluster.load.async.to.lds.b8 %src, %dst, 0, 0, %mask : !llvm.ptr<1>, !llvm.ptr<3>
  // CHECK: call void @llvm.amdgcn.cluster.load.async.to.lds.b32
  rocdl.cluster.load.async.to.lds.b32 %src, %dst, 0, 0, %mask : !llvm.ptr<1>, !llvm.ptr<3>
  // CHECK: call void @llvm.amdgcn.cluster.load.async.to.lds.b64
  rocdl.cluster.load.async.to.lds.b64 %src, %dst, 0, 0, %mask : !llvm.ptr<1>, !llvm.ptr<3>
  // CHECK: call void @llvm.amdgcn.cluster.load.async.to.lds.b128
  rocdl.cluster.load.async.to.lds.b128 %src, %dst, 0, 0, %mask : !llvm.ptr<1>, !llvm.ptr<3>
  llvm.return
}

// CHECK-LABEL: rocdl.tensor.load.to.lds
llvm.func @rocdl.tensor.load.to.lds(%dgroup0 : vector<4xi32>, %dgroup1 : vector<8xi32>,
                                    %dgroup2 : vector<4xi32>, %dgroup3 : vector<4xi32>) {
  // CHECK: call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %{{.*}}, <8 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 0)
  rocdl.tensor.load.to.lds %dgroup0, %dgroup1, %dgroup2, %dgroup3 cachepolicy 0 : vector<4xi32>, vector<8xi32>
  llvm.return
}

// CHECK-LABEL: rocdl.tensor.store.from.lds
llvm.func @rocdl.tensor.store.from.lds(%dgroup0 : vector<4xi32>, %dgroup1 : vector<8xi32>,
                                       %dgroup2 : vector<4xi32>, %dgroup3 : vector<4xi32>) {
  // CHECK: call void @llvm.amdgcn.tensor.store.from.lds(<4 x i32> %{{.*}}, <8 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 0)
  rocdl.tensor.store.from.lds %dgroup0, %dgroup1, %dgroup2, %dgroup3 cachepolicy 0 : vector<4xi32>, vector<8xi32>
  llvm.return
}

// CHECK-LABEL: rocdl.tensor.load.to.lds.d2
llvm.func @rocdl.tensor.load.to.lds.d2(%dgroup0 : vector<4xi32>, %dgroup1 : vector<8xi32>) {
  // CHECK: call void @llvm.amdgcn.tensor.load.to.lds.d2(<4 x i32> %{{.*}}, <8 x i32> %{{.*}}, i32 0)
  rocdl.tensor.load.to.lds.d2 %dgroup0, %dgroup1 cachepolicy 0 : vector<4xi32>, vector<8xi32>
  llvm.return
}

// CHECK-LABEL: rocdl.tensor.store.from.lds.d2
llvm.func @rocdl.tensor.store.from.lds.d2(%dgroup0 : vector<4xi32>, %dgroup1 : vector<8xi32>) {
  // CHECK: call void @llvm.amdgcn.tensor.store.from.lds.d2(<4 x i32> %{{.*}}, <8 x i32> %{{.*}}, i32 0)
  rocdl.tensor.store.from.lds.d2 %dgroup0, %dgroup1 cachepolicy 0 : vector<4xi32>, vector<8xi32>
  llvm.return
}

llvm.func @rocdl.make.buffer.rsrc(%ptr : !llvm.ptr,
                                  %stride : i16,
                                  %numRecords : i64,
                                  %flags : i32) -> !llvm.ptr<8> {
  // CHECK-LABEL: rocdl.make.buffer.rsrc
  // CHECK: %[[rsrc:.*]] = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %{{.*}}, i16 %{{.*}}, i64 %{{.*}}, i32 %{{.*}})
  // CHECK: ret ptr addrspace(8) %[[rsrc]]
  %rsrc = rocdl.make.buffer.rsrc %ptr, %stride, %numRecords, %flags : !llvm.ptr to !llvm.ptr<8>
  llvm.return %rsrc : !llvm.ptr<8>
}

llvm.func @rocdl.make.buffer.rsrc.p7.p1(%ptr : !llvm.ptr<1>,
                                  %stride : i16,
                                  %numRecords : i64,
                                  %flags : i32) -> !llvm.ptr<7> {
  // CHECK-LABEL: rocdl.make.buffer.rsrc.p7.p1
  // CHECK: %[[rsrc:.*]] = call ptr addrspace(7) @llvm.amdgcn.make.buffer.rsrc.p7.p1(ptr addrspace(1) %{{.*}}, i16 %{{.*}}, i64 %{{.*}}, i32 %{{.*}})
  // CHECK: ret ptr addrspace(7) %[[rsrc]]
  %rsrc = rocdl.make.buffer.rsrc %ptr, %stride, %numRecords, %flags : <1> to <7>
  llvm.return %rsrc : !llvm.ptr<7>
}

llvm.func @rocdl.permlanex16(%src0 : f32, %src1 : i32, %src2 : vector<2 x f32>, %src3 : vector<2 x i32>) -> f32 {
  %cst0 = llvm.mlir.constant(-1 : i32) : i32
  // CHECK-LABEL: rocdl.permlanex16
  // CHECK: call float @llvm.amdgcn.permlanex16.f32(float %{{.*}}, float %{{.*}}, i32 -1, i32 -1, i1 false, i1 true)
  %ret0 = rocdl.permlanex16 %src0, %src0, %cst0, %cst0, 0, -1 : f32, i32
  // CHECK: call i32 @llvm.amdgcn.permlanex16.i32(i32 %{{.*}}, i32 %{{.*}}, i32 -1, i32 -1, i1 false, i1 true)
  %ret1 = rocdl.permlanex16 %src1, %src1, %cst0, %cst0, 0, -1 : i32, i32
  // CHECK: call <2 x float> @llvm.amdgcn.permlanex16.v2f32(<2 x float> %{{.*}}, <2 x float> %{{.*}}, i32 -1, i32 -1, i1 false, i1 true)
  %ret2 = rocdl.permlanex16 %src2, %src2, %cst0, %cst0, 0, -1 : vector<2 x f32>, i32
  // CHECK: call <2 x i32> @llvm.amdgcn.permlanex16.v2i32(<2 x i32> %{{.*}}, <2 x i32> %{{.*}}, i32 -1, i32 -1, i1 false, i1 true)
  %ret3 = rocdl.permlanex16 %src3, %src3, %cst0, %cst0, 0, -1 : vector<2 x i32>, i32
  llvm.return %ret0 : f32
}

llvm.func @rocdl.permlane16.swap(%src : i32) -> !llvm.struct<(i32, i32)> {
  // CHECK-LABEL: rocdl.permlane16.swap
  // CHECK: call { i32, i32 } @llvm.amdgcn.permlane16.swap(i32 %{{.*}}, i32 %{{.*}}, i1 false, i1 true)
  %ret = rocdl.permlane16.swap %src, %src, 0, -1  : (i32, i32) -> !llvm.struct<(i32, i32)>
  llvm.return %ret : !llvm.struct<(i32, i32)>
}

llvm.func @rocdl.permlane32.swap(%src : i32) -> !llvm.struct<(i32, i32)> {
  // CHECK-LABEL: rocdl.permlane32.swap
  // CHECK: call { i32, i32 } @llvm.amdgcn.permlane32.swap(i32 %{{.*}}, i32 %{{.*}}, i1 false, i1 true)
  %ret = rocdl.permlane32.swap %src, %src, 0, -1  : (i32, i32) -> !llvm.struct<(i32, i32)>
  llvm.return %ret : !llvm.struct<(i32, i32)>
}

llvm.func @rocdl.wmma.fp8(%arg0 : vector<2 x i32>, %arg1 : vector<8xf32>) -> vector<8xf32> {
  // CHECK: call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.fp8.fp8.v8f32.v2i32(<2 x i32> %{{.*}}, <2 x i32> %{{.*}}, <8 x float> %{{.*}})
  %r0 = rocdl.wmma.f32.16x16x16.fp8_fp8 %arg0, %arg0, %arg1: (vector<2xi32>, vector<2xi32>, vector<8xf32>) -> vector<8xf32>

  // CHECK: call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.fp8.bf8.v8f32.v2i32(<2 x i32> %{{.*}}, <2 x i32> %{{.*}}, <8 x float> %{{.*}})
  %r1 = rocdl.wmma.f32.16x16x16.fp8_bf8 %arg0, %arg0, %arg1: (vector<2xi32>, vector<2xi32>, vector<8xf32>) -> vector<8xf32>

  // CHECK: call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.bf8.bf8.v8f32.v2i32(<2 x i32> %{{.*}}, <2 x i32> %{{.*}}, <8 x float> %{{.*}})
  %r2 = rocdl.wmma.f32.16x16x16.bf8_bf8 %arg0, %arg0, %arg1: (vector<2xi32>, vector<2xi32>, vector<8xf32>) -> vector<8xf32>

  // CHECK: call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.bf8.fp8.v8f32.v2i32(<2 x i32> %{{.*}}, <2 x i32> %{{.*}}, <8 x float> %{{.*}})
  %r3 = rocdl.wmma.f32.16x16x16.bf8_fp8 %arg0, %arg0, %arg1: (vector<2xi32>, vector<2xi32>, vector<8xf32>) -> vector<8xf32>

  llvm.return %r0 : vector<8 x f32>
}

llvm.func @rocdl.raw.ptr.buffer(%rsrc : !llvm.ptr<8>,
                        %offset : i32, %soffset : i32,
                        %vdata1 : i32,
                        %vdata2 : vector<2xi32>,
                        %vdata4 : vector<4xi32>) {
  %aux = llvm.mlir.constant(0 : i32) : i32
  // CHECK-LABEL: rocdl.raw.ptr.buffer
  // CHECK: call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 {{.*}}
  // CHECK: call <2 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v2i32(ptr addrspace(8) %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 {{.*}}
  // CHECK: call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 {{.*}}

  // CHECK: call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %{{.*}}, ptr addrspace(8) %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 {{.*}}
  // CHECK: call void @llvm.amdgcn.raw.ptr.buffer.store.v2i32(<2 x i32> %{{.*}}, ptr addrspace(8) %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 {{.*}}
  // CHECK: call void @llvm.amdgcn.raw.ptr.buffer.store.v4i32(<4 x i32> %{{.*}}, ptr addrspace(8) %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 {{.*}}

  %r1 = rocdl.raw.ptr.buffer.load %rsrc, %offset, %soffset, %aux : i32
  %r2 = rocdl.raw.ptr.buffer.load %rsrc, %offset, %soffset, %aux : vector<2xi32>
  %r4 = rocdl.raw.ptr.buffer.load %rsrc, %offset, %soffset, %aux : vector<4xi32>

  rocdl.raw.ptr.buffer.store %vdata1, %rsrc, %offset, %soffset, %aux : i32
  rocdl.raw.ptr.buffer.store %vdata2, %rsrc, %offset, %soffset, %aux : vector<2xi32>
  rocdl.raw.ptr.buffer.store %vdata4, %rsrc, %offset, %soffset, %aux : vector<4xi32>

  llvm.return
}

llvm.func @rocdl.raw.ptr.buffer.load.lds(%rsrc : !llvm.ptr<8>, %dstLds : !llvm.ptr<3>,
                        %voffset : i32, %soffset : i32) {
  %size = llvm.mlir.constant(4 : i32) : i32
  %offset = llvm.mlir.constant(128 : i32) : i32
  %aux = llvm.mlir.constant(1 : i32) : i32
  // CHECK-LABEL: rocdl.raw.ptr.buffer.load.lds
  // CHECK: call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) %{{.*}}, ptr addrspace(3) %{{.*}}, i32 4, i32 %{{.*}}, i32 %{{.*}}, i32 128, i32 1
  rocdl.raw.ptr.buffer.load.lds %rsrc, %dstLds, %size, %voffset, %soffset, %offset, %aux

  llvm.return
}

llvm.func @rocdl.global.prefetch(%ptr : !llvm.ptr<1>) {
  // CHECK-LABEL: rocdl.global.prefetch
  // CHECK: call void @llvm.amdgcn.global.prefetch(ptr addrspace(1) %{{.*}}, i32 0)
  rocdl.global.prefetch %ptr, scope 0 : !llvm.ptr<1>
  llvm.return
}

llvm.func @rocdl.flat.prefetch(%ptr : !llvm.ptr) {
  // CHECK-LABEL: rocdl.flat.prefetch
  // CHECK: call void @llvm.amdgcn.flat.prefetch(ptr %{{.*}}, i32 0)
  rocdl.flat.prefetch %ptr, scope 0 : !llvm.ptr
  llvm.return
}

llvm.func @rocdl.atomic.barriers.arrive(%ptr : !llvm.ptr<3>, %val : i64) {
  // CHECK-LABEL: rocdl.atomic.barriers.arrive
  // CHECK: call void @llvm.amdgcn.ds.atomic.async.barrier.arrive.b64(ptr addrspace(3) %{{.*}})
  // CHECK: %{{.*}} = call i64 @llvm.amdgcn.ds.atomic.barrier.arrive.rtn.b64(ptr addrspace(3) %{{.*}}, i64 %{{.*}})
  rocdl.ds.atomic.async.barrier.arrive.b64 %ptr : !llvm.ptr<3>
  %res = rocdl.ds.atomic.barrier.arrive.rtn.b64 %ptr, %val : !llvm.ptr<3>, i64 -> i64
  llvm.return
}

llvm.func @rocdl.wmma.scale(%arg0: i32, %arg1: vector<4xf32>, %arg2: vector<8xi32>,
                            %arg3: vector<12xi32>, %arg5: vector<16xi32>,
                            %arg8: i64, %arg9: vector<8xf32>) -> vector<4xf32> {
  // CHECK-LABEL: rocdl.wmma.scale

  // Test with default attributes (all zeros/false)
  // CHECK: call <4 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v4f32.v16i32.v16i32(i32 0, <16 x i32> %{{.*}}, i32 0, <16 x i32> %{{.*}}, i16 0, <4 x float> %{{.*}}, i32 0, i32 0, i32 %{{.*}}, i32 0, i32 0, i32 %{{.*}}, i1 false, i1 false)
  %r00 = rocdl.wmma.scale.f32.16x16x128.f8f6f4 %arg5, %arg5, %arg1, %arg0, %arg0
    {fmtA = 0 : i32, fmtB = 0 : i32, modC = 0 : i16,
     scaleAType = 0 : i32, fmtScaleA = 0 : i32,
     scaleBType = 0 : i32, fmtScaleB = 0 : i32,
     reuseA = false, reuseB = false} :
    (vector<16xi32>, vector<16xi32>, vector<4xf32>, i32, i32) -> vector<4xf32>

  // Test with different matrix formats (FP8 x BF8)
  // CHECK: call <4 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v4f32.v16i32.v16i32(i32 0, <16 x i32> %{{.*}}, i32 1, <16 x i32> %{{.*}}, i16 0, <4 x float> %{{.*}}, i32 1, i32 1, i32 %{{.*}}, i32 1, i32 1, i32 %{{.*}}, i1 false, i1 false)
  %r01 = rocdl.wmma.scale.f32.16x16x128.f8f6f4 %arg5, %arg5, %arg1, %arg0, %arg0
    {fmtA = 0 : i32, fmtB = 1 : i32, modC = 0 : i16,
     scaleAType = 1 : i32, fmtScaleA = 1 : i32,
     scaleBType = 1 : i32, fmtScaleB = 1 : i32,
     reuseA = false, reuseB = false} :
    (vector<16xi32>, vector<16xi32>, vector<4xf32>, i32, i32) -> vector<4xf32>

  // Test with FP8 x FP6 (different vector sizes) and modC = 1 (negate)
  // CHECK: call <4 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v4f32.v16i32.v12i32(i32 0, <16 x i32> %{{.*}}, i32 2, <12 x i32> %{{.*}}, i16 1, <4 x float> %{{.*}}, i32 2, i32 2, i32 %{{.*}}, i32 2, i32 2, i32 %{{.*}}, i1 false, i1 false)
  %r02 = rocdl.wmma.scale.f32.16x16x128.f8f6f4 %arg5, %arg3, %arg1, %arg0, %arg0
    {fmtA = 0 : i32, fmtB = 2 : i32, modC = 1 : i16,
     scaleAType = 2 : i32, fmtScaleA = 2 : i32,
     scaleBType = 2 : i32, fmtScaleB = 2 : i32,
     reuseA = false, reuseB = false} :
    (vector<16xi32>, vector<12xi32>, vector<4xf32>, i32, i32) -> vector<4xf32>

  // Test with BF8 x BF6 and modC = 2 (abs)
  // CHECK: call <4 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v4f32.v16i32.v12i32(i32 1, <16 x i32> %{{.*}}, i32 3, <12 x i32> %{{.*}}, i16 2, <4 x float> %{{.*}}, i32 0, i32 0, i32 %{{.*}}, i32 0, i32 0, i32 %{{.*}}, i1 false, i1 false)
  %r03 = rocdl.wmma.scale.f32.16x16x128.f8f6f4 %arg5, %arg3, %arg1, %arg0, %arg0
    {fmtA = 1 : i32, fmtB = 3 : i32, modC = 2 : i16,
     scaleAType = 0 : i32, fmtScaleA = 0 : i32,
     scaleBType = 0 : i32, fmtScaleB = 0 : i32,
     reuseA = false, reuseB = false} :
    (vector<16xi32>, vector<12xi32>, vector<4xf32>, i32, i32) -> vector<4xf32>

  // Test with FP8 x FP4 and modC = 3 (negate(abs))
  // CHECK: call <4 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v4f32.v16i32.v8i32(i32 0, <16 x i32> %{{.*}}, i32 4, <8 x i32> %{{.*}}, i16 3, <4 x float> %{{.*}}, i32 3, i32 3, i32 %{{.*}}, i32 3, i32 3, i32 %{{.*}}, i1 false, i1 false)
  %r04 = rocdl.wmma.scale.f32.16x16x128.f8f6f4 %arg5, %arg2, %arg1, %arg0, %arg0
    {fmtA = 0 : i32, fmtB = 4 : i32, modC = 3 : i16,
     scaleAType = 3 : i32, fmtScaleA = 3 : i32,
     scaleBType = 3 : i32, fmtScaleB = 3 : i32,
     reuseA = false, reuseB = false} :
    (vector<16xi32>, vector<8xi32>, vector<4xf32>, i32, i32) -> vector<4xf32>

  // Test with reuseA = true
  // CHECK: call <4 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v4f32.v16i32.v16i32(i32 2, <16 x i32> %{{.*}}, i32 2, <16 x i32> %{{.*}}, i16 0, <4 x float> %{{.*}}, i32 0, i32 0, i32 %{{.*}}, i32 0, i32 0, i32 %{{.*}}, i1 true, i1 false)
  %r10 = rocdl.wmma.scale.f32.16x16x128.f8f6f4 %arg5, %arg5, %arg1, %arg0, %arg0
    {fmtA = 2 : i32, fmtB = 2 : i32, modC = 0 : i16,
     scaleAType = 0 : i32, fmtScaleA = 0 : i32,
     scaleBType = 0 : i32, fmtScaleB = 0 : i32,
     reuseA = true, reuseB = false} :
    (vector<16xi32>, vector<16xi32>, vector<4xf32>, i32, i32) -> vector<4xf32>

  // Test with reuseB = true
  // CHECK: call <4 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v4f32.v16i32.v16i32(i32 3, <16 x i32> %{{.*}}, i32 3, <16 x i32> %{{.*}}, i16 0, <4 x float> %{{.*}}, i32 0, i32 0, i32 %{{.*}}, i32 0, i32 0, i32 %{{.*}}, i1 false, i1 true)
  %r11 = rocdl.wmma.scale.f32.16x16x128.f8f6f4 %arg5, %arg5, %arg1, %arg0, %arg0
    {fmtA = 3 : i32, fmtB = 3 : i32, modC = 0 : i16,
     scaleAType = 0 : i32, fmtScaleA = 0 : i32,
     scaleBType = 0 : i32, fmtScaleB = 0 : i32,
     reuseA = false, reuseB = true} :
    (vector<16xi32>, vector<16xi32>, vector<4xf32>, i32, i32) -> vector<4xf32>

  // Test with both reuseA and reuseB = true
  // CHECK: call <4 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v4f32.v16i32.v16i32(i32 4, <16 x i32> %{{.*}}, i32 4, <16 x i32> %{{.*}}, i16 1, <4 x float> %{{.*}}, i32 1, i32 1, i32 %{{.*}}, i32 1, i32 1, i32 %{{.*}}, i1 true, i1 true)
  %r12 = rocdl.wmma.scale.f32.16x16x128.f8f6f4 %arg5, %arg5, %arg1, %arg0, %arg0
    {fmtA = 4 : i32, fmtB = 4 : i32, modC = 1 : i16,
     scaleAType = 1 : i32, fmtScaleA = 1 : i32,
     scaleBType = 1 : i32, fmtScaleB = 1 : i32,
     reuseA = true, reuseB = true} :
    (vector<16xi32>, vector<16xi32>, vector<4xf32>, i32, i32) -> vector<4xf32>

  // Test scale16 variant with i64 scale exponents
  // CHECK: call <4 x float> @llvm.amdgcn.wmma.scale16.f32.16x16x128.f8f6f4.v4f32.v16i32.v16i32(i32 0, <16 x i32> %{{.*}}, i32 1, <16 x i32> %{{.*}}, i16 2, <4 x float> %{{.*}}, i32 2, i32 2, i64 %{{.*}}, i32 2, i32 2, i64 %{{.*}}, i1 false, i1 false)
  %r_scale16 = rocdl.wmma.scale16.f32.16x16x128.f8f6f4 %arg5, %arg5, %arg1, %arg8, %arg8
    {fmtA = 0 : i32, fmtB = 1 : i32, modC = 2 : i16,
     scaleAType = 2 : i32, fmtScaleA = 2 : i32,
     scaleBType = 2 : i32, fmtScaleB = 2 : i32,
     reuseA = false, reuseB = false} :
    (vector<16xi32>, vector<16xi32>, vector<4xf32>, i64, i64) -> vector<4xf32>

  // Test f4 variant (no matrix format parameters)
  // CHECK: call <8 x float> @llvm.amdgcn.wmma.scale.f32.32x16x128.f4.v8f32.v16i32.v8i32(<16 x i32> %{{.*}}, <8 x i32> %{{.*}}, i16 0, <8 x float> %{{.*}}, i32 1, i32 1, i32 %{{.*}}, i32 1, i32 1, i32 %{{.*}}, i1 false, i1 false)
  %r_f4 = rocdl.wmma.scale.f32.32x16x128.f4 %arg5, %arg2, %arg9, %arg0, %arg0
    {modC = 0 : i16,
     scaleAType = 1 : i32, fmtScaleA = 1 : i32,
     scaleBType = 1 : i32, fmtScaleB = 1 : i32,
     reuseA = false, reuseB = false} :
    (vector<16xi32>, vector<8xi32>, vector<8xf32>, i32, i32) -> vector<8xf32>

  // Test f4 scale16 variant with varied attributes
  // CHECK: call <8 x float> @llvm.amdgcn.wmma.scale16.f32.32x16x128.f4.v8f32.v16i32.v8i32(<16 x i32> %{{.*}}, <8 x i32> %{{.*}}, i16 3, <8 x float> %{{.*}}, i32 2, i32 3, i64 %{{.*}}, i32 3, i32 2, i64 %{{.*}}, i1 true, i1 true)
  %r_f4_scale16 = rocdl.wmma.scale16.f32.32x16x128.f4 %arg5, %arg2, %arg9, %arg8, %arg8
    {modC = 3 : i16,
     scaleAType = 2 : i32, fmtScaleA = 3 : i32,
     scaleBType = 3 : i32, fmtScaleB = 2 : i32,
     reuseA = true, reuseB = true} :
    (vector<16xi32>, vector<8xi32>, vector<8xf32>, i64, i64) -> vector<8xf32>

  llvm.return %r00 : vector<4xf32>
}

llvm.func @rocdl.raw.ptr.buffer.atomic.f32(%rsrc : !llvm.ptr<8>,
                        %offset : i32, %soffset : i32,
                        %vdata1 : f32) {
  %aux = llvm.mlir.constant(0 : i32) : i32
  // CHECK-LABEL: rocdl.raw.ptr.buffer.atomic.f32
  // CHECK: call float @llvm.amdgcn.raw.ptr.buffer.atomic.fadd.f32(float %{{.*}}, ptr addrspace(8) %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 {{.*}}
  // CHECK: call float @llvm.amdgcn.raw.ptr.buffer.atomic.fmax.f32(float %{{.*}}, ptr addrspace(8) %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 {{.*}}

  rocdl.raw.ptr.buffer.atomic.fadd %vdata1, %rsrc, %offset, %soffset, %aux : f32
  rocdl.raw.ptr.buffer.atomic.fmax %vdata1, %rsrc, %offset, %soffset, %aux : f32

  llvm.return
}

llvm.func @rocdl.raw.ptr.buffer.atomic.i32(%rsrc : !llvm.ptr<8>,
                        %offset : i32, %soffset : i32,
                        %vdata1 : i32) {
  %aux = llvm.mlir.constant(0 : i32) : i32
  // CHECK-LABEL: rocdl.raw.ptr.buffer.atomic.i32
  // CHECK: call i32 @llvm.amdgcn.raw.ptr.buffer.atomic.smax.i32(i32 %{{.*}}, ptr addrspace(8) %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 {{.*}}
  // CHECK: call i32 @llvm.amdgcn.raw.ptr.buffer.atomic.umin.i32(i32 %{{.*}}, ptr addrspace(8) %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 {{.*}}

  rocdl.raw.ptr.buffer.atomic.smax %vdata1, %rsrc, %offset, %soffset, %aux : i32
  rocdl.raw.ptr.buffer.atomic.umin %vdata1, %rsrc, %offset, %soffset, %aux : i32

  llvm.return
}

llvm.func @rocdl.raw.ptr.buffer.atomic.cmpswap(%rsrc : !llvm.ptr<8>,
                        %offset : i32, %soffset : i32,
                        %src : i32, %cmp : i32) -> i32 {
  %aux = llvm.mlir.constant(0 : i32) : i32
  // CHECK-LABEL: rocdl.raw.ptr.buffer.atomic.cmpswap
  // CHECK: [[val:%.+]] = call i32 @llvm.amdgcn.raw.ptr.buffer.atomic.cmpswap.i32(i32 %{{.*}}, i32 %{{.*}}, ptr addrspace(8) %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 {{.*}}
  // CHECK: ret i32 [[val]]

  %val = rocdl.raw.ptr.buffer.atomic.cmpswap %src, %cmp, %rsrc, %offset, %soffset, %aux : i32
  llvm.return %val : i32
}

llvm.func @rocdl.raw.buffer(%rsrc : vector<4xi32>,
                        %offset : i32, %soffset : i32,
                        %vdata1 : i32,
                        %vdata2 : vector<2xi32>,
                        %vdata4 : vector<4xi32>) {
  %aux = llvm.mlir.constant(0 : i32) : i32
  // CHECK-LABEL: rocdl.raw.buffer
  // CHECK: call i32 @llvm.amdgcn.raw.buffer.load.i32(<4 x i32> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 {{.*}}
  // CHECK: call <2 x i32> @llvm.amdgcn.raw.buffer.load.v2i32(<4 x i32> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 {{.*}}
  // CHECK: call <4 x i32> @llvm.amdgcn.raw.buffer.load.v4i32(<4 x i32> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 {{.*}}

  // CHECK: call void @llvm.amdgcn.raw.buffer.store.i32(i32 %{{.*}}, <4 x i32> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 {{.*}}
  // CHECK: call void @llvm.amdgcn.raw.buffer.store.v2i32(<2 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 {{.*}}
  // CHECK: call void @llvm.amdgcn.raw.buffer.store.v4i32(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 {{.*}}

  %r1 = rocdl.raw.buffer.load %rsrc, %offset, %soffset, %aux : i32
  %r2 = rocdl.raw.buffer.load %rsrc, %offset, %soffset, %aux : vector<2xi32>
  %r4 = rocdl.raw.buffer.load %rsrc, %offset, %soffset, %aux : vector<4xi32>

  rocdl.raw.buffer.store %vdata1, %rsrc, %offset, %soffset, %aux : i32
  rocdl.raw.buffer.store %vdata2, %rsrc, %offset, %soffset, %aux : vector<2xi32>
  rocdl.raw.buffer.store %vdata4, %rsrc, %offset, %soffset, %aux : vector<4xi32>

  llvm.return
}

llvm.func @rocdl.raw.buffer.atomic.f32(%rsrc : vector<4xi32>,
                        %offset : i32, %soffset : i32,
                        %vdata1 : f32) {
  %aux = llvm.mlir.constant(0 : i32) : i32
  // CHECK-LABEL: rocdl.raw.buffer.atomic.f32
  // CHECK: call float @llvm.amdgcn.raw.buffer.atomic.fadd.f32(float %{{.*}}, <4 x i32> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 {{.*}}
  // CHECK: call float @llvm.amdgcn.raw.buffer.atomic.fmax.f32(float %{{.*}}, <4 x i32> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 {{.*}}

  rocdl.raw.buffer.atomic.fadd %vdata1, %rsrc, %offset, %soffset, %aux : f32
  rocdl.raw.buffer.atomic.fmax %vdata1, %rsrc, %offset, %soffset, %aux : f32

  llvm.return
}

llvm.func @rocdl.raw.buffer.atomic.i32(%rsrc : vector<4xi32>,
                        %offset : i32, %soffset : i32,
                        %vdata1 : i32) {
  %aux = llvm.mlir.constant(0 : i32) : i32
  // CHECK-LABEL: rocdl.raw.buffer.atomic.i32
  // CHECK: call i32 @llvm.amdgcn.raw.buffer.atomic.smax.i32(i32 %{{.*}}, <4 x i32> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 {{.*}}
  // CHECK: call i32 @llvm.amdgcn.raw.buffer.atomic.umin.i32(i32 %{{.*}}, <4 x i32> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 {{.*}}

  rocdl.raw.buffer.atomic.smax %vdata1, %rsrc, %offset, %soffset, %aux : i32
  rocdl.raw.buffer.atomic.umin %vdata1, %rsrc, %offset, %soffset, %aux : i32

  llvm.return
}

llvm.func @rocdl.raw.buffer.atomic.cmpswap(%rsrc : vector<4xi32>,
                        %offset : i32, %soffset : i32,
                        %src : i32, %cmp : i32) -> i32 {
  %aux = llvm.mlir.constant(0 : i32) : i32
  // CHECK-LABEL: rocdl.raw.buffer.atomic.cmpswap
  // CHECK: [[val:%.+]] = call i32 @llvm.amdgcn.raw.buffer.atomic.cmpswap.i32(i32 %{{.*}}, i32 %{{.*}}, <4 x i32> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 {{.*}}
  // CHECK: ret i32 [[val]]

  %val = rocdl.raw.buffer.atomic.cmpswap(%src, %cmp, %rsrc, %offset, %soffset, %aux) : i32, vector<4xi32>
  llvm.return %val : i32
}

llvm.func @rocdl_8bit_floats(%source: i32, %source_half: f16, %source_bfloat: bf16, %source_packed: vector<2xf16>, %stoch: i32) -> i32 {
// CHECK-LABEL: @rocdl_8bit_floats
// CHECK: call float @llvm.amdgcn.cvt.f32.bf8(i32 %{{.+}}, i32 0)
// CHECK: call float @llvm.amdgcn.cvt.scalef32.f32.bf8(i32 %{{.+}}, float 1.000000e+00, i32 0)
// CHECK: call float @llvm.amdgcn.cvt.f32.fp8(i32 %{{.+}}, i32 0)
// CHECK: call float @llvm.amdgcn.cvt.scalef32.f32.fp8(i32 %{{.+}}, float 1.000000e+00, i32 0)
// CHECK: call <2 x half> @llvm.amdgcn.cvt.scalef32.pk.f16.bf8(i32 %{{.+}}, float 1.000000e+00, i1 false)
// CHECK: call <2 x half> @llvm.amdgcn.cvt.scalef32.pk.f16.fp8(i32 %{{.+}}, float 1.000000e+00, i1 false)
// CHECK: call <2 x half> @llvm.amdgcn.cvt.scalef32.f16.fp8(<2 x half> %{{.+}}, i32 %{{.+}}, float 1.000000e+00, i32 0, i1 false)
// CHECK: call <2 x half> @llvm.amdgcn.cvt.scalef32.f16.bf8(<2 x half> %{{.+}}, i32 %{{.+}}, float 1.000000e+00, i32 0, i1 false)
// CHECK: call <2 x bfloat> @llvm.amdgcn.cvt.scalef32.pk.bf16.bf8(i32 %{{.+}}, float 1.000000e+00, i1 false)
// CHECK: call <2 x bfloat> @llvm.amdgcn.cvt.scalef32.pk.bf16.fp8(i32 %{{.+}}, float 1.000000e+00, i1 false)
// CHECK: call i32 @llvm.amdgcn.cvt.pk.bf8.f32(float %{{.+}}, float %{{.+}}, i32 %{{.+}}, i1 false)
// CHECK: call i32 @llvm.amdgcn.cvt.pk.fp8.f32(float %{{.+}}, float %{{.+}}, i32 %{{.+}}, i1 false)
// CHECK: call i32 @llvm.amdgcn.cvt.sr.bf8.f32(float %{{.+}}, i32 %{{.+}}, i32 %{{.+}}, i32 2)
// CHECK: call i32 @llvm.amdgcn.cvt.sr.fp8.f32(float %{{.+}}, i32 %{{.+}}, i32 %{{.+}}, i32 3)
// CHECK: call i32 @llvm.amdgcn.cvt.scalef32.sr.fp8.f32(i32 %{{.+}}, float %{{.+}}, i32 %{{.+}}, float 1.000000e+00, i32 3)
// CHECK: call i32 @llvm.amdgcn.cvt.scalef32.sr.fp8.f16(i32 %{{.+}}, half %{{.+}}, i32 %{{.+}}, float 1.000000e+00, i32 3)
// CHECK: call i32 @llvm.amdgcn.cvt.scalef32.sr.fp8.bf16(i32 %{{.+}}, bfloat %{{.+}}, i32 %{{.+}}, float 1.000000e+00, i32 3)
// CHECK: call i32 @llvm.amdgcn.cvt.sr.bf8.f32(float %{{.+}}, i32 %{{.+}}, i32 %{{.+}}, i32 3)
// CHECK: call i32 @llvm.amdgcn.cvt.scalef32.sr.bf8.f32(i32 %{{.+}}, float %{{.+}}, i32 %{{.+}}, float 1.000000e+00, i32 3)
// CHECK: call i32 @llvm.amdgcn.cvt.scalef32.sr.bf8.f16(i32 %{{.+}}, half %{{.+}}, i32 %{{.+}}, float 1.000000e+00, i32 3)
// CHECK: call i32 @llvm.amdgcn.cvt.scalef32.sr.bf8.bf16(i32 %{{.+}}, bfloat %{{.+}}, i32 %{{.+}}, float 1.000000e+00, i32 3)
// CHECK: call <2 x float> @llvm.amdgcn.cvt.scalef32.pk.f32.fp8(i32 %{{.+}}, float 1.000000e+00, i1 false)
// CHECK: call <2 x float> @llvm.amdgcn.cvt.scalef32.pk.f32.bf8(i32 %{{.+}}, float 1.000000e+00, i1 false)
  %c4 = llvm.mlir.constant(1.0 : f32) : f32
  %false = llvm.mlir.constant(false) : i1
  %v1 = rocdl.cvt.f32.bf8 %source[0] : f32
  %v1_scaled = rocdl.cvt.scalef32.f32.bf8 %source[0], %c4 : f32
  %v2 = rocdl.cvt.f32.fp8 %source[0] : f32
  %v2_scaled = rocdl.cvt.scalef32.f32.fp8 %source[0], %c4 : f32
  %v3_scaled = rocdl.cvt.scalef32.pk.f16.bf8 %source[false], %c4 : vector<2xf16>
  %v4_scaled = rocdl.cvt.scalef32.pk.f16.fp8 %source[false], %c4 : vector<2xf16>
  %v5 = rocdl.cvt.scalef32.f16.fp8 %source[0], %c4 -> %source_packed[false] : vector<2xf16>
  %v6 = rocdl.cvt.scalef32.f16.bf8 %source[0], %c4 -> %source_packed[false] : vector<2xf16>
  %v7 = rocdl.cvt.scalef32.pk.bf16.bf8 %source[false], %c4 : vector<2xbf16>
  %v8 = rocdl.cvt.scalef32.pk.bf16.fp8 %source[false], %c4 : vector<2xbf16>
  %source2 = rocdl.cvt.pk.bf8.f32 %v1, %v2 -> %source[false] : i32
  %source3 = rocdl.cvt.pk.fp8.f32 %v1, %v2 -> %source2[false] : i32
  %source4 = rocdl.cvt.sr.bf8.f32 %v1, %stoch -> %source3[2] : i32
  %source5 = rocdl.cvt.sr.fp8.f32 %v2, %stoch -> %source4[3] : i32
  %source5_scaled = rocdl.cvt.scalef32.sr.fp8.f32 %v2, %stoch, %c4 -> %source4[3] : i32
  %source5_scaled_half = rocdl.cvt.scalef32.sr.fp8.f16 %source_half, %stoch, %c4 -> %source4[3] : i32
  %source5_scaled_bfloat = rocdl.cvt.scalef32.sr.fp8.bf16 %source_bfloat, %stoch, %c4 -> %source4[3] : i32
  %source6 = rocdl.cvt.sr.bf8.f32 %v1, %stoch -> %source3[3] : i32
  %source6_scaled  = rocdl.cvt.scalef32.sr.bf8.f32 %v2, %stoch, %c4 -> %source3[3] : i32
  %source6_scaled_half = rocdl.cvt.scalef32.sr.bf8.f16 %source_half, %stoch, %c4 -> %source3[3] : i32
  %source6_scaled_bfloat = rocdl.cvt.scalef32.sr.bf8.bf16 %source_bfloat, %stoch, %c4 -> %source3[3] : i32
  %source7_scaled = rocdl.cvt.scalef32.pk.f32.fp8 %source[false], %c4 : vector<2xf32>
  %source8_scaled = rocdl.cvt.scalef32.pk.f32.bf8 %source[false], %c4 : vector<2xf32>
  llvm.return %source5 : i32
}

llvm.func @rocdl_8bit_packed_v2i16(%sourceA: f32, %sourceB: f32, %old: vector<2xi16>) -> vector<2xi16> {
// CHECK-LABEL:  @rocdl_8bit_packed_v2i16
// CHECK: call <2 x i16> @llvm.amdgcn.cvt.scalef32.pk.fp8.f32(<2 x i16> %{{.+}}, float %{{.+}}, float %{{.+}}, float 1.000000e+00, i1 false)
// CHECK: call <2 x i16> @llvm.amdgcn.cvt.scalef32.pk.bf8.f32(<2 x i16> %{{.+}}, float %{{.+}}, float %{{.+}}, float 1.000000e+00, i1 false)
  %c0 = llvm.mlir.constant(1.0 : f32) : f32
  %source_scaled = rocdl.cvt.scalef32.pk.fp8.f32 %sourceA, %sourceB, %c0 -> %old[false] : vector<2xi16>
  %source2_scaled = rocdl.cvt.scalef32.pk.bf8.f32 %sourceA, %sourceB, %c0 -> %old[false] : vector<2xi16>
  llvm.return %source_scaled : vector<2xi16>
}

llvm.func @rocdl_v2f16_v2i16(%source: vector<2xf16>, %source2: vector<2xbf16>, %old: vector<2xi16>) -> vector<2xi16> {
// CHECK-LABEL: @rocdl_v2f16_v2i16
// CHECK: call <2 x i16> @llvm.amdgcn.cvt.scalef32.pk.fp8.f16(<2 x i16> %2, <2 x half> %0, float 1.000000e+00, i1 false)
// CHECK: call <2 x i16> @llvm.amdgcn.cvt.scalef32.pk.fp8.bf16(<2 x i16> %2, <2 x bfloat> %1, float 1.000000e+00, i1 false)
// CHECK: call <2 x i16> @llvm.amdgcn.cvt.scalef32.pk.bf8.f16(<2 x i16> %2, <2 x half> %0, float 1.000000e+00, i1 false)
// CHECK: call <2 x i16> @llvm.amdgcn.cvt.scalef32.pk.bf8.bf16(<2 x i16> %2, <2 x bfloat> %1, float 1.000000e+00, i1 false)
  %c0 = llvm.mlir.constant(1.0 : f32) : f32
  %source_scaled = rocdl.cvt.scalef32.pk.fp8.f16 %source, %c0 -> %old[false] : vector<2xi16>
  %source2_scaled = rocdl.cvt.scalef32.pk.fp8.bf16 %source2, %c0 -> %old[false] : vector<2xi16>
  %source3_scaled = rocdl.cvt.scalef32.pk.bf8.f16 %source, %c0 -> %old[false] : vector<2xi16>
  %source4_scaled = rocdl.cvt.scalef32.pk.bf8.bf16 %source2, %c0 -> %old[false] : vector<2xi16>
  llvm.return %source_scaled : vector<2xi16>
}

llvm.func @rocdl_16bit_packed_floats(%sourceA: f32, %sourceB: f32) -> vector<2xf16> {
  // CHECK-LABEL: @rocdl_16bit_packed_floats
  // CHECK: call <2 x half> @llvm.amdgcn.cvt.pkrtz(float {{.*}}, float {{.*}})
  %source = rocdl.cvt.pkrtz %sourceA, %sourceB  : vector<2xf16>
  llvm.return %source : vector<2xf16>
}

// CHECK-LABEL: @rocdl_6_bit_floats
// CHECK-SAME: (<6 x i32> %[[V32F6:.+]], <16 x float> %[[V16F32:.+]], <32 x float> %[[V32F32:.+]], <32 x half> %[[V32F16:.+]], <32 x bfloat> %[[V32BF16:.+]], i32 %[[SEED:.+]], float %[[SCALE:.+]])
llvm.func @rocdl_6_bit_floats(
    %v32f6: vector<6xi32>, %v16f32: vector<16xf32>, %v32f32: vector<32xf32>,
    %v32f16: vector<32xf16>, %v32bf16: vector<32xbf16>, %seed: i32,
    %scale: f32) {
  // CHECK-NEXT: call <6 x i32> @llvm.amdgcn.cvt.scalef32.2xpk16.bf6.f32(<16 x float> %[[V16F32]], <16 x float> %[[V16F32]], float %[[SCALE]])
  %f32_to_bf6 = rocdl.cvt.scalef32.2xpk16.bf6.f32 %v16f32, %v16f32, %scale : vector<6xi32>
  // CHECK-NEXT: call <6 x i32> @llvm.amdgcn.cvt.scalef32.2xpk16.fp6.f32(<16 x float> %[[V16F32]], <16 x float> %[[V16F32]], float %[[SCALE]])
  %f32_to_fp6 = rocdl.cvt.scalef32.2xpk16.fp6.f32 %v16f32, %v16f32, %scale : vector<6xi32>
  // CHECK-NEXT: call <6 x i32> @llvm.amdgcn.cvt.scalef32.pk32.bf6.f16(<32 x half> %[[V32F16]], float %[[SCALE]])
  %f16_to_bf6 = rocdl.cvt.scalef32.pk32.bf6.f16 %v32f16, %scale : vector<6xi32>
  // CHECK-NEXT: call <6 x i32> @llvm.amdgcn.cvt.scalef32.pk32.fp6.f16(<32 x half> %[[V32F16]], float %[[SCALE]])
  %f16_to_fp6 = rocdl.cvt.scalef32.pk32.fp6.f16 %v32f16, %scale : vector<6xi32>
  // CHECK-NEXT: call <6 x i32> @llvm.amdgcn.cvt.scalef32.pk32.bf6.bf16(<32 x bfloat> %[[V32BF16]], float %[[SCALE]])
  %bf16_to_bf6 = rocdl.cvt.scalef32.pk32.bf6.bf16 %v32bf16, %scale : vector<6xi32>
  // CHECK-NEXT: call <6 x i32> @llvm.amdgcn.cvt.scalef32.pk32.fp6.bf16(<32 x bfloat> %[[V32BF16]], float %[[SCALE]])
  %bf16_to_fp6 = rocdl.cvt.scalef32.pk32.fp6.bf16 %v32bf16, %scale : vector<6xi32>

  // CHECK-NEXT: call <32 x float> @llvm.amdgcn.cvt.scalef32.pk32.f32.bf6(<6 x i32> %[[V32F6]], float %[[SCALE]])
  %bf6_to_f32 = rocdl.cvt.scalef32.pk32.f32.bf6 %v32f6, %scale : vector<32xf32>
  // CHECK-NEXT: call <32 x float> @llvm.amdgcn.cvt.scalef32.pk32.f32.fp6(<6 x i32> %[[V32F6]], float %[[SCALE]])
  %fp6_to_f32 = rocdl.cvt.scalef32.pk32.f32.fp6 %v32f6, %scale : vector<32xf32>
  // CHECK-NEXT: call <32 x half> @llvm.amdgcn.cvt.scalef32.pk32.f16.bf6(<6 x i32> %[[V32F6]], float %[[SCALE]])
  %bf6_to_f16 = rocdl.cvt.scalef32.pk32.f16.bf6 %v32f6, %scale : vector<32xf16>
  // CHECK-NEXT: call <32 x half> @llvm.amdgcn.cvt.scalef32.pk32.f16.fp6(<6 x i32> %[[V32F6]], float %[[SCALE]])
  %fp6_to_f16 = rocdl.cvt.scalef32.pk32.f16.fp6 %v32f6, %scale : vector<32xf16>
  // CHECK-NEXT: call <32 x bfloat> @llvm.amdgcn.cvt.scalef32.pk32.bf16.bf6(<6 x i32> %[[V32F6]], float %[[SCALE]])
  %bf6_to_bf16 = rocdl.cvt.scalef32.pk32.bf16.bf6 %v32f6, %scale : vector<32xbf16>
  // CHECK-NEXT: call <32 x bfloat> @llvm.amdgcn.cvt.scalef32.pk32.bf16.fp6(<6 x i32> %[[V32F6]], float %[[SCALE]])
  %fp6_to_bf16 = rocdl.cvt.scalef32.pk32.bf16.fp6 %v32f6, %scale : vector<32xbf16>

  // CHECK-NEXT: call <6 x i32> @llvm.amdgcn.cvt.scalef32.sr.pk32.bf6.f32(<32 x float> %[[V32F32]], i32 %[[SEED]], float %[[SCALE]])
  %f32_to_bf6_sr = rocdl.cvt.scalef32.sr.pk32.bf6.f32 %v32f32, %seed, %scale : vector<6xi32>
  // CHECK-NEXT: call <6 x i32> @llvm.amdgcn.cvt.scalef32.sr.pk32.fp6.f32(<32 x float> %[[V32F32]], i32 %[[SEED]], float %[[SCALE]])
  %f32_to_fp6_sr = rocdl.cvt.scalef32.sr.pk32.fp6.f32 %v32f32, %seed, %scale : vector<6xi32>
  // CHECK-NEXT: call <6 x i32> @llvm.amdgcn.cvt.scalef32.sr.pk32.bf6.f16(<32 x half> %[[V32F16]], i32 %[[SEED]], float %[[SCALE]])
  %f16_to_bf6_sr = rocdl.cvt.scalef32.sr.pk32.bf6.f16 %v32f16, %seed, %scale : vector<6xi32>
  // CHECK-NEXT: call <6 x i32> @llvm.amdgcn.cvt.scalef32.sr.pk32.fp6.f16(<32 x half> %[[V32F16]], i32 %[[SEED]], float %[[SCALE]])
  %f16_to_fp6_sr = rocdl.cvt.scalef32.sr.pk32.fp6.f16 %v32f16, %seed, %scale : vector<6xi32>
  // CHECK-NEXT: call <6 x i32> @llvm.amdgcn.cvt.scalef32.sr.pk32.bf6.bf16(<32 x bfloat> %[[V32BF16]], i32 %[[SEED]], float %[[SCALE]])
  %bf16_to_bf6_sr = rocdl.cvt.scalef32.sr.pk32.bf6.bf16 %v32bf16, %seed, %scale : vector<6xi32>
  // CHECK-NEXT: call <6 x i32> @llvm.amdgcn.cvt.scalef32.sr.pk32.fp6.bf16(<32 x bfloat> %[[V32BF16]], i32 %[[SEED]], float %[[SCALE]])
  %bf16_to_fp6_sr = rocdl.cvt.scalef32.sr.pk32.fp6.bf16 %v32bf16, %seed, %scale : vector<6xi32>

  llvm.return
}

// CHECK-LABEL: @rocdl_4_bit_floats
// CHECK-SAME: (i32 %[[V8F4:.+]], float %[[F32:.+]], <2 x float> %[[V2F32:.+]], <2 x half> %[[V2F16:.+]], <2 x bfloat> %[[V2BF16:.+]], i32 %[[SEED:.+]], float %[[SCALE:.+]])
llvm.func @rocdl_4_bit_floats(
    %v8f4: i32, %f32: f32, %v2f32: vector<2xf32>, %v2f16: vector<2xf16>,
    %v2bf16: vector<2xbf16>, %seed: i32, %scale: f32) {

  // CHECK-NEXT: call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 %[[V8F4]], float %[[F32]], float %[[F32]], float %[[SCALE]], i32 0)
  %f32_to_fp4 = rocdl.cvt.scalef32.pk.fp4.f32 %f32, %f32, %scale -> %v8f4[0] : i32
  // CHECK-NEXT: call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f16(i32 %[[V8F4]], <2 x half> %[[V2F16]], float %[[SCALE]], i32 1)
  %f16_to_fp4 = rocdl.cvt.scalef32.pk.fp4.f16 %v2f16, %scale -> %v8f4[1] : i32
  // CHECK-NEXT: call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.bf16(i32 %[[V8F4]], <2 x bfloat> %[[V2BF16]], float %[[SCALE]], i32 0)
  %bf16_to_fp4 = rocdl.cvt.scalef32.pk.fp4.bf16 %v2bf16, %scale -> %v8f4[0] : i32

  // CHECK-NEXT: call <2 x float> @llvm.amdgcn.cvt.scalef32.pk.f32.fp4(i32 %[[V8F4]], float %[[SCALE]], i32 0)
  %fp4_to_f32 = rocdl.cvt.scalef32.pk.f32.fp4 %v8f4[0], %scale : vector<2xf32>
  // CHECK-NEXT: call <2 x half> @llvm.amdgcn.cvt.scalef32.pk.f16.fp4(i32 %[[V8F4]], float %[[SCALE]], i32 1)
  %fp4_to_f16 = rocdl.cvt.scalef32.pk.f16.fp4 %v8f4[1], %scale : vector<2xf16>
  // CHECK-NEXT: call <2 x bfloat> @llvm.amdgcn.cvt.scalef32.pk.bf16.fp4(i32 %[[V8F4]], float %[[SCALE]], i32 0)
  %fp4_to_bf16 = rocdl.cvt.scalef32.pk.bf16.fp4 %v8f4[0], %scale : vector<2xbf16>

  // CHECK-NEXT: call i32 @llvm.amdgcn.cvt.scalef32.sr.pk.fp4.f32(i32 %[[V8F4]], <2 x float> %[[V2F32]], i32 %[[SEED]], float %[[SCALE]], i32 0)
  %f32_to_fp4_sr = rocdl.cvt.scalef32.sr.pk.fp4.f32 %v2f32, %seed, %scale -> %v8f4[0] : i32
  // CHECK-NEXT: call i32 @llvm.amdgcn.cvt.scalef32.sr.pk.fp4.f16(i32 %[[V8F4]], <2 x half> %[[V2F16]], i32 %[[SEED]], float %[[SCALE]], i32 1)
  %f16_to_fp4_sr = rocdl.cvt.scalef32.sr.pk.fp4.f16 %v2f16, %seed, %scale -> %v8f4[1] : i32
  // CHECK-NEXT: call i32 @llvm.amdgcn.cvt.scalef32.sr.pk.fp4.bf16(i32 %[[V8F4]], <2 x bfloat> %[[V2BF16]], i32 %[[SEED]], float %[[SCALE]], i32 0)
  %bf16_to_fp4_sr = rocdl.cvt.scalef32.sr.pk.fp4.bf16 %v2bf16, %seed, %scale -> %v8f4[0] : i32

  llvm.return
}
llvm.func @rocdl_atomic_attrs(%ptr: !llvm.ptr<1>, %data: f32) {
  // CHECK-LABEL: @rocdl_atomic_attrs
  // CHECK: atomicrmw
  // CHECK-SAME: !amdgpu.ignore.denormal.mode
  // CHECK-SAME: !amdgpu.no.fine.grained.memory
  // CHECK-SAME: !amdgpu.no.remote.memory
  llvm.atomicrmw fadd %ptr, %data monotonic {
    rocdl.ignore_denormal_mode,
    rocdl.no_fine_grained_memory,
    rocdl.no_remote_memory} : !llvm.ptr<1>, f32
  llvm.return
}

llvm.func @rocdl_last_use(%ptr: !llvm.ptr<1>) -> i32 {
  // CHECK-LABEL: @rocdl_last_use
  // CHECK: %[[ret:.+]] = load
  // CHECK-SAME: !amdgpu.last.use
  // CHECK: ret i32 %[[ret]]
  %ret = llvm.load %ptr {rocdl.last_use} : !llvm.ptr<1> -> i32
  llvm.return %ret : i32
}

llvm.func @test_fmed3_f16(%arg0: f16, %arg1: f16, %arg2: f16) -> f16 {
  // CHECK-LABEL: define half @test_fmed3_f16(half %0, half %1, half %2)
  %0 = rocdl.fmed3 %arg0, %arg1, %arg2 : f16
  llvm.return %0 : f16
  // CHECK: call half @llvm.amdgcn.fmed3.f16(half %0, half %1, half %2)
}

llvm.func @test_fmed3_f32(%arg0: f32, %arg1: f32, %arg2: f32) -> f32 {
  // CHECK-LABEL: define float @test_fmed3_f32(float %0, float %1, float %2)
  %0 = rocdl.fmed3 %arg0, %arg1, %arg2 : f32
  llvm.return %0 : f32
  // CHECK: call float @llvm.amdgcn.fmed3.f32(float %0, float %1, float %2)
}

// CHECK-LABEL: rocdl.cvt.scale.pk8
// CHECK-SAME:(i32 %[[I32:.+]], <2 x i32> %[[V2I32:.+]], i32 %[[SCALE:.+]])
llvm.func @rocdl.cvt.scale.pk8(%i32: i32, %v2xi32: vector<2xi32>, %scale: i32) {

  // CHECK: call <8 x half> @llvm.amdgcn.cvt.scale.pk8.f16.fp4(i32 %[[I32]], i32 %[[SCALE]], i32 0)
  %0 = rocdl.cvt.scale.pk8.f16.fp4 %i32, %scale[0] : vector<8xf16>
  // CHECK: call <8 x bfloat> @llvm.amdgcn.cvt.scale.pk8.bf16.fp4(i32 %[[I32]], i32 %[[SCALE]], i32 0)
  %1 = rocdl.cvt.scale.pk8.bf16.fp4 %i32, %scale[0] : vector<8xbf16>
  // CHECK: call <8 x float> @llvm.amdgcn.cvt.scale.pk8.f32.fp4(i32 %[[I32]], i32 %[[SCALE]], i32 0)
  %2 = rocdl.cvt.scale.pk8.f32.fp4 %i32, %scale[0] : vector<8xf32>

  // CHECK: call <8 x half> @llvm.amdgcn.cvt.scale.pk8.f16.fp8(<2 x i32> %[[V2I32]], i32 %[[SCALE]], i32 0)
  %3 = rocdl.cvt.scale.pk8.f16.fp8 %v2xi32, %scale[0] : vector<8xf16>
  // CHECK: call <8 x bfloat> @llvm.amdgcn.cvt.scale.pk8.bf16.fp8(<2 x i32> %[[V2I32]], i32 %[[SCALE]], i32 0)
  %4 = rocdl.cvt.scale.pk8.bf16.fp8 %v2xi32, %scale[0] : vector<8xbf16>
  // CHECK: call <8 x float> @llvm.amdgcn.cvt.scale.pk8.f32.fp8(<2 x i32> %[[V2I32]], i32 %[[SCALE]], i32 0)
  %5 = rocdl.cvt.scale.pk8.f32.fp8 %v2xi32, %scale[0] : vector<8xf32>

  // CHECK: call <8 x half> @llvm.amdgcn.cvt.scale.pk8.f16.bf8(<2 x i32> %[[V2I32]], i32 %[[SCALE]], i32 0)
  %6 = rocdl.cvt.scale.pk8.f16.bf8 %v2xi32, %scale[0] : vector<8xf16>
  // CHECK: call <8 x bfloat> @llvm.amdgcn.cvt.scale.pk8.bf16.bf8(<2 x i32> %[[V2I32]], i32 %[[SCALE]], i32 0)
  %7 = rocdl.cvt.scale.pk8.bf16.bf8 %v2xi32, %scale[0] : vector<8xbf16>
  // CHECK: call <8 x float> @llvm.amdgcn.cvt.scale.pk8.f32.bf8(<2 x i32> %[[V2I32]], i32 %[[SCALE]], i32 0)
  %8 = rocdl.cvt.scale.pk8.f32.bf8 %v2xi32, %scale[0] : vector<8xf32>

  llvm.return
}

// CHECK-LABEL: rocdl.cvt.scalef32.pk8
// CHECK-SAME:(<8 x float> %[[V8F32:.+]], <8 x half> %[[V8F16:.+]], <8 x bfloat> %[[V8BF16:.+]], float %[[SCALE:.+]])
llvm.func @rocdl.cvt.scalef32.pk8(%v8xf32: vector<8xf32>, %v8xf16: vector<8xf16>, %v8xbf16: vector<8xbf16>, %scale: f32) {

  // CHECK: call <2 x i32> @llvm.amdgcn.cvt.scalef32.pk8.fp8.f32(<8 x float> %[[V8F32]], float %[[SCALE]])
  %0 = rocdl.cvt.scalef32.pk8.fp8.f32 %v8xf32, %scale : vector<2xi32>
  // CHECK: call <2 x i32> @llvm.amdgcn.cvt.scalef32.pk8.bf8.f32(<8 x float> %[[V8F32]], float %[[SCALE]])
  %1 = rocdl.cvt.scalef32.pk8.bf8.f32 %v8xf32, %scale : vector<2xi32>
  // CHECK: call i32 @llvm.amdgcn.cvt.scalef32.pk8.fp4.f32(<8 x float> %[[V8F32]], float %[[SCALE]])
  %2 = rocdl.cvt.scalef32.pk8.fp4.f32 %v8xf32, %scale : i32

  // CHECK: call <2 x i32> @llvm.amdgcn.cvt.scalef32.pk8.fp8.f16(<8 x half> %[[V8F16]], float %[[SCALE]])
  %3 = rocdl.cvt.scalef32.pk8.fp8.f16 %v8xf16, %scale : vector<2xi32>
  // CHECK: call <2 x i32> @llvm.amdgcn.cvt.scalef32.pk8.bf8.f16(<8 x half> %[[V8F16]], float %[[SCALE]])
  %4 = rocdl.cvt.scalef32.pk8.bf8.f16 %v8xf16, %scale : vector<2xi32>
  // CHECK: call i32 @llvm.amdgcn.cvt.scalef32.pk8.fp4.f16(<8 x half> %[[V8F16]], float %[[SCALE]])
  %5 = rocdl.cvt.scalef32.pk8.fp4.f16 %v8xf16, %scale : i32

  // CHECK: call <2 x i32> @llvm.amdgcn.cvt.scalef32.pk8.fp8.bf16(<8 x bfloat> %[[V8BF16]], float %[[SCALE]])
  %6 = rocdl.cvt.scalef32.pk8.fp8.bf16 %v8xbf16, %scale : vector<2xi32>
  // CHECK: call <2 x i32> @llvm.amdgcn.cvt.scalef32.pk8.bf8.bf16(<8 x bfloat> %[[V8BF16]], float %[[SCALE]])
  %7 = rocdl.cvt.scalef32.pk8.bf8.bf16 %v8xbf16, %scale : vector<2xi32>
  // CHECK: call i32 @llvm.amdgcn.cvt.scalef32.pk8.fp4.bf16(<8 x bfloat> %[[V8BF16]], float %[[SCALE]])
  %8 = rocdl.cvt.scalef32.pk8.fp4.bf16 %v8xbf16, %scale : i32

  llvm.return
}

// CHECK-LABEL: rocdl.cvt.scalef32.sr.pk8
// CHECK-SAME:(<8 x float> %[[V8F32:.+]], <8 x half> %[[V8F16:.+]], <8 x bfloat> %[[V8BF16:.+]], i32 %[[SEED:.+]],  float %[[SCALE:.+]])
llvm.func @rocdl.cvt.scalef32.sr.pk8(%v8xf32: vector<8xf32>,
                                     %v8xf16: vector<8xf16>,
                                     %v8xbf16: vector<8xbf16>,
                                     %seed: i32,
                                     %scale: f32) {

  // CHECK: call <2 x i32> @llvm.amdgcn.cvt.scalef32.sr.pk8.fp8.f32(<8 x float> %[[V8F32]], i32 %[[SEED]], float %[[SCALE]])
  %0 = rocdl.cvt.scalef32.sr.pk8.fp8.f32 %v8xf32, %seed, %scale : vector<2xi32>
  // CHECK: call <2 x i32> @llvm.amdgcn.cvt.scalef32.sr.pk8.bf8.f32(<8 x float> %[[V8F32]], i32 %[[SEED]], float %[[SCALE]])
  %1 = rocdl.cvt.scalef32.sr.pk8.bf8.f32 %v8xf32, %seed, %scale : vector<2xi32>
  // CHECK: call i32 @llvm.amdgcn.cvt.scalef32.sr.pk8.fp4.f32(<8 x float> %[[V8F32]], i32 %[[SEED]], float %[[SCALE]])
  %2 = rocdl.cvt.scalef32.sr.pk8.fp4.f32 %v8xf32, %seed, %scale : i32

  // CHECK: call <2 x i32> @llvm.amdgcn.cvt.scalef32.sr.pk8.fp8.f16(<8 x half> %[[V8F16]], i32 %[[SEED]], float %[[SCALE]])
  %3 = rocdl.cvt.scalef32.sr.pk8.fp8.f16 %v8xf16, %seed, %scale : vector<2xi32>
  // CHECK: call <2 x i32> @llvm.amdgcn.cvt.scalef32.sr.pk8.bf8.f16(<8 x half> %[[V8F16]], i32 %[[SEED]], float %[[SCALE]])
  %4 = rocdl.cvt.scalef32.sr.pk8.bf8.f16 %v8xf16, %seed, %scale : vector<2xi32>
  // CHECK: call i32 @llvm.amdgcn.cvt.scalef32.sr.pk8.fp4.f16(<8 x half> %[[V8F16]], i32 %[[SEED]], float %[[SCALE]])
  %5 = rocdl.cvt.scalef32.sr.pk8.fp4.f16 %v8xf16, %seed, %scale : i32

  // CHECK: call <2 x i32> @llvm.amdgcn.cvt.scalef32.sr.pk8.fp8.bf16(<8 x bfloat> %[[V8BF16]], i32 %[[SEED]], float %[[SCALE]])
  %6 = rocdl.cvt.scalef32.sr.pk8.fp8.bf16 %v8xbf16, %seed, %scale : vector<2xi32>
  // CHECK: call <2 x i32> @llvm.amdgcn.cvt.scalef32.sr.pk8.bf8.bf16(<8 x bfloat> %[[V8BF16]], i32 %[[SEED]], float %[[SCALE]])
  %7 = rocdl.cvt.scalef32.sr.pk8.bf8.bf16 %v8xbf16, %seed, %scale : vector<2xi32>
  // CHECK: call i32 @llvm.amdgcn.cvt.scalef32.sr.pk8.fp4.bf16(<8 x bfloat> %[[V8BF16]], i32 %[[SEED]], float %[[SCALE]])
  %8 = rocdl.cvt.scalef32.sr.pk8.fp4.bf16 %v8xbf16, %seed, %scale : i32

  llvm.return
}


// CHECK-LABEL: @rocdl.cvt.scale.pk16
// CHECK-SAME:(<3 x i32> %[[SRC0:.+]], i32 %[[SCALE:.+]])
llvm.func @rocdl.cvt.scale.pk16(%v3xi32: vector<3xi32>, %scale:i32) {

  // CHECK: call <16 x half> @llvm.amdgcn.cvt.scale.pk16.f16.fp6(<3 x i32> %[[SRC0]], i32 %[[SCALE]], i32 0)
  %0 = rocdl.cvt.scale.pk16.f16.fp6 %v3xi32, %scale[0] : vector<16xf16>
  // CHECK: call <16 x bfloat> @llvm.amdgcn.cvt.scale.pk16.bf16.fp6(<3 x i32> %[[SRC0]], i32 %[[SCALE]], i32 0)
  %1 = rocdl.cvt.scale.pk16.bf16.fp6 %v3xi32, %scale[0] : vector<16xbf16>
  // CHECK: call <16 x float> @llvm.amdgcn.cvt.scale.pk16.f32.fp6(<3 x i32> %[[SRC0]], i32 %[[SCALE]], i32 0)
  %2 = rocdl.cvt.scale.pk16.f32.fp6 %v3xi32, %scale[0] : vector<16xf32>
  // CHECK: call <16 x half> @llvm.amdgcn.cvt.scale.pk16.f16.bf6(<3 x i32> %[[SRC0]], i32 %[[SCALE]], i32 0)
  %3 = rocdl.cvt.scale.pk16.f16.bf6 %v3xi32, %scale[0] : vector<16xf16>
  // CHECK: call <16 x bfloat> @llvm.amdgcn.cvt.scale.pk16.bf16.bf6(<3 x i32> %[[SRC0]], i32 %[[SCALE]], i32 0)
  %4 = rocdl.cvt.scale.pk16.bf16.bf6 %v3xi32, %scale[0] : vector<16xbf16>
  // CHECK:  call <16 x float> @llvm.amdgcn.cvt.scale.pk16.f32.bf6(<3 x i32> %[[SRC0]], i32 %[[SCALE]], i32 0)
  %5 = rocdl.cvt.scale.pk16.f32.bf6 %v3xi32, %scale[0] : vector<16xf32>

  llvm.return
}

// CHECK-LABEL: rocdl.cvt.scalef32.pk16
// CHECK-SAME:(<16 x float> %[[V16F32:.+]], <16 x half> %[[V16F16:.+]], <16 x bfloat> %[[V16BF16:.+]], float %[[SCALE:.+]])
llvm.func @rocdl.cvt.scalef32.pk16(%v16xf32: vector<16xf32>, %v16xf16: vector<16xf16>, %v16xbf16: vector<16xbf16>, %scale: f32) {

  // CHECK: call <3 x i32> @llvm.amdgcn.cvt.scalef32.pk16.fp6.f16(<16 x half> %[[V16F16]], float %[[SCALE]])
  %0 = rocdl.cvt.scalef32.pk16.fp6.f16 %v16xf16, %scale : vector<3xi32>
  // CHECK: call <3 x i32> @llvm.amdgcn.cvt.scalef32.pk16.fp6.bf16(<16 x bfloat> %[[V16BF16]], float %[[SCALE]])
  %1 = rocdl.cvt.scalef32.pk16.fp6.bf16 %v16xbf16, %scale : vector<3xi32>
  // CHECK: call <3 x i32> @llvm.amdgcn.cvt.scalef32.pk16.fp6.f32(<16 x float> %[[V16F32]], float %[[SCALE]])
  %2 = rocdl.cvt.scalef32.pk16.fp6.f32 %v16xf32, %scale : vector<3xi32>

  // CHECK: call <3 x i32> @llvm.amdgcn.cvt.scalef32.pk16.bf6.f16(<16 x half> %[[V16F16]], float %[[SCALE]])
  %3 = rocdl.cvt.scalef32.pk16.bf6.f16 %v16xf16, %scale : vector<3xi32>
  // CHECK: call <3 x i32> @llvm.amdgcn.cvt.scalef32.pk16.bf6.bf16(<16 x bfloat> %[[V16BF16]], float %[[SCALE]])
  %4 = rocdl.cvt.scalef32.pk16.bf6.bf16 %v16xbf16, %scale : vector<3xi32>
  // CHECK: call <3 x i32> @llvm.amdgcn.cvt.scalef32.pk16.bf6.f32(<16 x float> %[[V16F32]], float %[[SCALE]])
  %5 = rocdl.cvt.scalef32.pk16.bf6.f32 %v16xf32, %scale : vector<3xi32>

  llvm.return
}

// CHECK-LABEL: rocdl.cvt.scalef32.sr.pk16
// CHECK-SAME:(<16 x float> %[[V16F32:.+]], <16 x half> %[[V16F16:.+]], <16 x bfloat> %[[V16BF16:.+]], i32 %[[SEED:.+]],  float %[[SCALE:.+]])
llvm.func @rocdl.cvt.scalef32.sr.pk16(%v16xf32: vector<16xf32>,
                                     %v16xf16: vector<16xf16>,
                                     %v16xbf16: vector<16xbf16>,
                                     %seed: i32,
                                     %scale: f32) {

  // CHECK: call <3 x i32> @llvm.amdgcn.cvt.scalef32.sr.pk16.fp6.f16(<16 x half> %[[V16F16]], i32 %[[SEED]], float %[[SCALE]])
  %0 = rocdl.cvt.scalef32.sr.pk16.fp6.f16 %v16xf16, %seed, %scale : vector<3xi32>
  // CHECK: call <3 x i32> @llvm.amdgcn.cvt.scalef32.sr.pk16.fp6.bf16(<16 x bfloat> %[[V16BF16]], i32 %[[SEED]], float %[[SCALE]])
  %1 = rocdl.cvt.scalef32.sr.pk16.fp6.bf16 %v16xbf16, %seed, %scale : vector<3xi32>
  // CHECK: call <3 x i32> @llvm.amdgcn.cvt.scalef32.sr.pk16.fp6.f32(<16 x float> %[[V16F32]], i32 %[[SEED]], float %[[SCALE]])
  %2 = rocdl.cvt.scalef32.sr.pk16.fp6.f32 %v16xf32, %seed, %scale : vector<3xi32>

  // CHECK: call <3 x i32> @llvm.amdgcn.cvt.scalef32.sr.pk16.bf6.f16(<16 x half> %[[V16F16]], i32 %[[SEED]], float %[[SCALE]])
  %3 = rocdl.cvt.scalef32.sr.pk16.bf6.f16 %v16xf16, %seed, %scale : vector<3xi32>
  // CHECK: call <3 x i32> @llvm.amdgcn.cvt.scalef32.sr.pk16.bf6.bf16(<16 x bfloat> %[[V16BF16]], i32 %[[SEED]], float %[[SCALE]])
  %4 = rocdl.cvt.scalef32.sr.pk16.bf6.bf16 %v16xbf16, %seed, %scale : vector<3xi32>
  // CHECK: call <3 x i32> @llvm.amdgcn.cvt.scalef32.sr.pk16.bf6.f32(<16 x float> %[[V16F32]], i32 %[[SEED]], float %[[SCALE]])
  %5 = rocdl.cvt.scalef32.sr.pk16.bf6.f32 %v16xf32, %seed, %scale : vector<3xi32>

  llvm.return
}

// CHECK-DAG: attributes #[[$KERNEL_ATTRS]] = { "amdgpu-flat-work-group-size"="1,256" "uniform-work-group-size"="true" }
// CHECK-DAG: attributes #[[$KERNEL_WORKGROUP_ATTRS]] = { "amdgpu-flat-work-group-size"="1,1024"
// CHECK-DAG: attributes #[[$KNOWN_BLOCK_SIZE_ATTRS]] = { "amdgpu-flat-work-group-size"="128,128"
// CHECK-DAG: attributes #[[$KERNEL_NO_UNIFORM_WORK_GROUPS_ATTRS]] = { "amdgpu-flat-work-group-size"="1,256" "uniform-work-group-size"="false" }
// CHECK-DAG: ![[$REQD_WORK_GROUP_SIZE]] = !{i32 16, i32 4, i32 2}
// CHECK-DAG: attributes #[[$KERNEL_WAVES_PER_EU_ATTR]] = { "amdgpu-flat-work-group-size"="1,256" "amdgpu-waves-per-eu"="2" "uniform-work-group-size"="true" }
// CHECK-DAG: attributes #[[$KERNEL_UNSAFE_FP_ATOMICS_ATTR]] = { "amdgpu-flat-work-group-size"="1,256" "amdgpu-unsafe-fp-atomics"="true" "uniform-work-group-size"="true" }
