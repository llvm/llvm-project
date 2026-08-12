// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: @llvm_nvvm_tcgen05_commit_generic
llvm.func @llvm_nvvm_tcgen05_commit_generic(%barrier : !llvm.ptr,
                                            %cta_mask : i16,
                                            %cta_mask_32 : i32) {
  // CHECK: call void @llvm.nvvm.tcgen05.commit.cg1.p0(ptr %{{.*}})
  nvvm.tcgen05.commit %barrier : !llvm.ptr

  // CHECK: call void @llvm.nvvm.tcgen05.commit.cg2.p0(ptr %{{.*}})
  nvvm.tcgen05.commit %barrier {group = #nvvm.cta_group<cta_2>} : !llvm.ptr

  // CHECK: call void @llvm.nvvm.tcgen05.commit.mc.cg1.p0.i16(ptr %{{.*}}, i16 %{{.*}})
  nvvm.tcgen05.commit %barrier, multicast_mask = %cta_mask : !llvm.ptr, i16

  // CHECK: call void @llvm.nvvm.tcgen05.commit.mc.cg2.p0.i16(ptr %{{.*}}, i16 %{{.*}})
  nvvm.tcgen05.commit %barrier, multicast_mask = %cta_mask {group = #nvvm.cta_group<cta_2>} : !llvm.ptr, i16

  // CHECK: call void @llvm.nvvm.tcgen05.commit.mc.cg1.p0.i32(ptr %{{.*}}, i32 %{{.*}})
  nvvm.tcgen05.commit %barrier, multicast_mask = %cta_mask_32 : !llvm.ptr, i32

  // CHECK: call void @llvm.nvvm.tcgen05.commit.mc.cg2.p0.i32(ptr %{{.*}}, i32 %{{.*}})
  nvvm.tcgen05.commit %barrier, multicast_mask = %cta_mask_32 {group = #nvvm.cta_group<cta_2>} : !llvm.ptr, i32
  llvm.return
}

// CHECK-LABEL: @llvm_nvvm_tcgen05_commit_shared
llvm.func @llvm_nvvm_tcgen05_commit_shared(%barrier : !llvm.ptr<3>,
                                           %cta_mask : i16,
                                           %cta_mask_32 : i32) {
  // CHECK: call void @llvm.nvvm.tcgen05.commit.cg1.p3(ptr addrspace(3) %{{.*}})
  nvvm.tcgen05.commit %barrier : !llvm.ptr<3>

  // CHECK: call void @llvm.nvvm.tcgen05.commit.cg2.p3(ptr addrspace(3) %{{.*}})
  nvvm.tcgen05.commit %barrier {group = #nvvm.cta_group<cta_2>} : !llvm.ptr<3>

  // CHECK: call void @llvm.nvvm.tcgen05.commit.mc.cg1.p3.i16(ptr addrspace(3) %{{.*}}, i16 %{{.*}})
  nvvm.tcgen05.commit %barrier, multicast_mask = %cta_mask : !llvm.ptr<3>, i16

  // CHECK: call void @llvm.nvvm.tcgen05.commit.mc.cg2.p3.i16(ptr addrspace(3) %{{.*}}, i16 %{{.*}})
  nvvm.tcgen05.commit %barrier, multicast_mask = %cta_mask {group = #nvvm.cta_group<cta_2>} : !llvm.ptr<3>, i16

  // CHECK: call void @llvm.nvvm.tcgen05.commit.mc.cg1.p3.i32(ptr addrspace(3) %{{.*}}, i32 %{{.*}})
  nvvm.tcgen05.commit %barrier, multicast_mask = %cta_mask_32 : !llvm.ptr<3>, i32

  // CHECK: call void @llvm.nvvm.tcgen05.commit.mc.cg2.p3.i32(ptr addrspace(3) %{{.*}}, i32 %{{.*}})
  nvvm.tcgen05.commit %barrier, multicast_mask = %cta_mask_32 {group = #nvvm.cta_group<cta_2>} : !llvm.ptr<3>, i32
  llvm.return
}
