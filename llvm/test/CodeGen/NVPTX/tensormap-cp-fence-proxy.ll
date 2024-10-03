; RUN: llc < %s -march=nvptx64 -mcpu=sm_90 -mattr=+ptx83 | FileCheck --check-prefixes=CHECK %s

; CHECK-LABEL: test_tensormap_cp_fenceproxy
define void @test_tensormap_cp_fenceproxy(ptr addrspace(1) %gptr, ptr addrspace(3) %sptr) {

  ; CHECK: tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.cta.sync.aligned [{{%rd[0-9]+}}], [{{%rd[0-9]+}}], 128;
  call void @llvm.nvvm.tensormap.cp_fenceproxy.global.shared.tensormap_generic.release.cta.sync.aligned(ptr addrspace(1) %gptr, ptr addrspace(3) %sptr, i32 128)

  ; CHECK: tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.cluster.sync.aligned [{{%rd[0-9]+}}], [{{%rd[0-9]+}}], 128;
  call void @llvm.nvvm.tensormap.cp_fenceproxy.global.shared.tensormap_generic.release.cluster.sync.aligned(ptr addrspace(1) %gptr, ptr addrspace(3) %sptr, i32 128)

  ; CHECK: tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned [{{%rd[0-9]+}}], [{{%rd[0-9]+}}], 128;
  call void @llvm.nvvm.tensormap.cp_fenceproxy.global.shared.tensormap_generic.release.gpu.sync.aligned(ptr addrspace(1) %gptr, ptr addrspace(3) %sptr, i32 128)

  ; CHECK: tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.sys.sync.aligned [{{%rd[0-9]+}}], [{{%rd[0-9]+}}], 128;
  call void @llvm.nvvm.tensormap.cp_fenceproxy.global.shared.tensormap_generic.release.sys.sync.aligned(ptr addrspace(1) %gptr, ptr addrspace(3) %sptr, i32 128)

  ret void
}