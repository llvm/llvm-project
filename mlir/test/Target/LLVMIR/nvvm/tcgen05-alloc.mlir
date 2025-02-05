// RUN: mlir-opt -split-input-file -verify-diagnostics %s
// RUN: mlir-translate -mlir-to-llvmir -split-input-file -verify-diagnostics %s | FileCheck %s --check-prefix=CHECK-LLVM

// CHECK-LABEL: @llvm_nvvm_tcgen05_alloc
llvm.func @llvm_nvvm_tcgen05_alloc(%addr : !llvm.ptr, %ncols : i32) {
  // CHECK-LLVM: call void @llvm.nvvm.tcgen05.alloc.cg1(ptr %{{.*}}, i32 %{{.*}})
  nvvm.tcgen05.alloc %addr, %ncols : !llvm.ptr, i32

  // CHECK-LLVM: call void @llvm.nvvm.tcgen05.alloc.cg2(ptr %{{.*}}, i32 %{{.*}})
  nvvm.tcgen05.alloc %addr, %ncols {group = #nvvm.tcgen05_group<cta_2>} : !llvm.ptr, i32
  llvm.return
}

// CHECK-LABEL: @llvm_nvvm_tcgen05_alloc_shared
llvm.func @llvm_nvvm_tcgen05_alloc_shared(%addr : !llvm.ptr<3>, %ncols : i32) {
  // CHECK-LLVM: call void @llvm.nvvm.tcgen05.alloc.shared.cg1(ptr addrspace(3) %{{.*}}, i32 %{{.*}})
  nvvm.tcgen05.alloc %addr, %ncols : !llvm.ptr<3>, i32

  // CHECK-LLVM: call void @llvm.nvvm.tcgen05.alloc.shared.cg2(ptr addrspace(3) %{{.*}}, i32 %{{.*}})
  nvvm.tcgen05.alloc %addr, %ncols {group = #nvvm.tcgen05_group<cta_2>} : !llvm.ptr<3>, i32
  llvm.return
}

// CHECK-LABEL: @llvm_nvvm_tcgen05_dealloc
llvm.func @llvm_nvvm_tcgen05_dealloc(%addr : !llvm.ptr<6>, %ncols : i32) {
  // CHECK-LLVM: call void @llvm.nvvm.tcgen05.dealloc.cg1(ptr addrspace(6) %{{.*}}, i32 %{{.*}})
  nvvm.tcgen05.dealloc %addr, %ncols : !llvm.ptr<6>, i32

  // CHECK-LLVM: call void @llvm.nvvm.tcgen05.dealloc.cg2(ptr addrspace(6) %{{.*}}, i32 %{{.*}})
  nvvm.tcgen05.dealloc %addr, %ncols {group = #nvvm.tcgen05_group<cta_2>} : !llvm.ptr<6>, i32
  llvm.return
}

// CHECK-LABEL: @llvm_nvvm_tcgen05_relinquish_alloc_permit
llvm.func @llvm_nvvm_tcgen05_relinquish_alloc_permit() {
  // CHECK-LLVM: call void @llvm.nvvm.tcgen05.relinq.alloc.permit.cg1()
  nvvm.tcgen05.relinquish_alloc_permit

  // CHECK-LLVM: call void @llvm.nvvm.tcgen05.relinq.alloc.permit.cg2()
  nvvm.tcgen05.relinquish_alloc_permit {group = #nvvm.tcgen05_group<cta_2>}
  llvm.return
}
