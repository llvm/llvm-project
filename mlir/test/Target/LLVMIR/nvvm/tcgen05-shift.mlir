// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: @llvm_nvvm_tcgen05_shift
llvm.func @llvm_nvvm_tcgen05_shift(%taddr : !llvm.ptr<6>) {
  // CHECK: call void @llvm.nvvm.tcgen05.shift.down.cg1(ptr addrspace(6) %{{.*}})
  nvvm.tcgen05.shift %taddr : !llvm.ptr<6>

  // CHECK: call void @llvm.nvvm.tcgen05.shift.down.cg2(ptr addrspace(6) %{{.*}})
  nvvm.tcgen05.shift %taddr {group = #nvvm.cta_group<cta_2>} : !llvm.ptr<6>
  llvm.return
}
