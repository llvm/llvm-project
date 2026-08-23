// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: @llvm_nvvm_tcgen05_commit_generic_smem_a_read
llvm.func @llvm_nvvm_tcgen05_commit_generic_smem_a_read(%barrier : !llvm.ptr,
                                                        %cta_mask : i16,
                                                        %cta_mask_32 : i32) {
  // CHECK: call void @llvm.nvvm.tcgen05.commit.smem.a.read.cg1.p0(ptr %{{.*}})
  nvvm.tcgen05.commit %barrier  smem_a_read = true : !llvm.ptr

  // CHECK: call void @llvm.nvvm.tcgen05.commit.smem.a.read.cg2.p0(ptr %{{.*}})
  nvvm.tcgen05.commit %barrier  group = <cta_2> smem_a_read = true : !llvm.ptr

  // CHECK: call void @llvm.nvvm.tcgen05.commit.smem.a.read.mc.cg1.p0.i16(ptr %{{.*}}, i16 %{{.*}})
  nvvm.tcgen05.commit %barrier, multicast_mask = %cta_mask  smem_a_read = true : !llvm.ptr, i16

  // CHECK: call void @llvm.nvvm.tcgen05.commit.smem.a.read.mc.cg2.p0.i16(ptr %{{.*}}, i16 %{{.*}})
  nvvm.tcgen05.commit %barrier, multicast_mask = %cta_mask  group = <cta_2> smem_a_read = true : !llvm.ptr, i16

  // CHECK: call void @llvm.nvvm.tcgen05.commit.smem.a.read.mc.cg1.p0.i32(ptr %{{.*}}, i32 %{{.*}})
  nvvm.tcgen05.commit %barrier, multicast_mask = %cta_mask_32  smem_a_read = true : !llvm.ptr, i32

  // CHECK: call void @llvm.nvvm.tcgen05.commit.smem.a.read.mc.cg2.p0.i32(ptr %{{.*}}, i32 %{{.*}})
  nvvm.tcgen05.commit %barrier, multicast_mask = %cta_mask_32  group = <cta_2> smem_a_read = true : !llvm.ptr, i32
  llvm.return
}

// CHECK-LABEL: @llvm_nvvm_tcgen05_commit_shared_smem_a_read
llvm.func @llvm_nvvm_tcgen05_commit_shared_smem_a_read(%barrier : !llvm.ptr<3>,
                                                       %cta_mask : i16,
                                                       %cta_mask_32 : i32) {
  // CHECK: call void @llvm.nvvm.tcgen05.commit.smem.a.read.cg1.p3(ptr addrspace(3) %{{.*}})
  nvvm.tcgen05.commit %barrier  smem_a_read = true : !llvm.ptr<3>

  // CHECK: call void @llvm.nvvm.tcgen05.commit.smem.a.read.cg2.p3(ptr addrspace(3) %{{.*}})
  nvvm.tcgen05.commit %barrier  group = <cta_2> smem_a_read = true : !llvm.ptr<3>

  // CHECK: call void @llvm.nvvm.tcgen05.commit.smem.a.read.mc.cg1.p3.i16(ptr addrspace(3) %{{.*}}, i16 %{{.*}})
  nvvm.tcgen05.commit %barrier, multicast_mask = %cta_mask  smem_a_read = true : !llvm.ptr<3>, i16

  // CHECK: call void @llvm.nvvm.tcgen05.commit.smem.a.read.mc.cg2.p3.i16(ptr addrspace(3) %{{.*}}, i16 %{{.*}})
  nvvm.tcgen05.commit %barrier, multicast_mask = %cta_mask  group = <cta_2> smem_a_read = true : !llvm.ptr<3>, i16

  // CHECK: call void @llvm.nvvm.tcgen05.commit.smem.a.read.mc.cg1.p3.i32(ptr addrspace(3) %{{.*}}, i32 %{{.*}})
  nvvm.tcgen05.commit %barrier, multicast_mask = %cta_mask_32  smem_a_read = true : !llvm.ptr<3>, i32

  // CHECK: call void @llvm.nvvm.tcgen05.commit.smem.a.read.mc.cg2.p3.i32(ptr addrspace(3) %{{.*}}, i32 %{{.*}})
  nvvm.tcgen05.commit %barrier, multicast_mask = %cta_mask_32  group = <cta_2> smem_a_read = true : !llvm.ptr<3>, i32
  llvm.return
}
