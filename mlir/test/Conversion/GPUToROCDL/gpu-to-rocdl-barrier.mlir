// RUN: mlir-opt %s -convert-gpu-to-rocdl='chipset=gfx950' --mlir-print-local-scope | FileCheck %s --check-prefixes=CHECK,GFX9
// RUN: mlir-opt %s -convert-gpu-to-rocdl='chipset=gfx1201' --mlir-print-local-scope | FileCheck %s --check-prefixes=CHECK,GFX12

gpu.module @test_module {
// CHECK-LABEL: func @barrier_default()
func.func @barrier_default() {
  // CHECK: llvm.fence syncscope("workgroup") release{{$}}
  // GFX9-NEXT: rocdl.s.barrier
  // GFX12-NEXT: rocdl.s.barrier.signal id = -1
  // GFX12-NEXT: rocdl.s.barrier.wait id = -1
  // CHECK-NEXT: llvm.fence syncscope("workgroup") acquire{{$}}
  gpu.barrier
  func.return
}

// CHECK-LABEL: func @barrier_no_fence()
func.func @barrier_no_fence() {
  // GFX9-NEXT: rocdl.s.barrier
  // GFX12-NEXT: rocdl.s.barrier.signal id = -1
  // GFX12-NEXT: rocdl.s.barrier.wait id = -1
  // CHECK-NOT: llvm.fence
  gpu.barrier memfence []
  func.return
}

// CHECK-LABEL: func @barrier_workgroup_only()
func.func @barrier_workgroup_only() {
  // CHECK: llvm.fence syncscope("workgroup") release {llvm.mmra = #llvm.mmra_tag<"amdgpu-synchronize-as":"local">}
  // GFX9-NEXT: rocdl.s.barrier
  // GFX12-NEXT: rocdl.s.barrier.signal id = -1
  // GFX12-NEXT: rocdl.s.barrier.wait id = -1
  // CHECK: llvm.fence syncscope("workgroup") acquire {llvm.mmra = #llvm.mmra_tag<"amdgpu-synchronize-as":"local">}
  gpu.barrier memfence [#gpu.address_space<workgroup>]
  func.return
}

// CHECK-LABEL: func @barrier_global_only()
func.func @barrier_global_only() {
  // CHECK-NEXT: llvm.fence syncscope("workgroup") release {llvm.mmra = #llvm.mmra_tag<"amdgpu-synchronize-as":"global">}
  // GFX9-NEXT: rocdl.s.barrier
  // GFX12-NEXT: rocdl.s.barrier.signal id = -1
  // GFX12-NEXT: rocdl.s.barrier.wait id = -1
  // CHECK-NEXT: llvm.fence syncscope("workgroup") acquire {llvm.mmra = #llvm.mmra_tag<"amdgpu-synchronize-as":"global">}
  gpu.barrier memfence [#gpu.address_space<global>]
  func.return
}

// CHECK-LABEL: func @barrier_both()
func.func @barrier_both() {
  // CHECK-NEXT: llvm.fence syncscope("workgroup") release{{$}}
  // GFX9-NEXT: rocdl.s.barrier
  // GFX12-NEXT: rocdl.s.barrier.signal id = -1
  // GFX12-NEXT: rocdl.s.barrier.wait id = -1
  // CHECK-NEXT: llvm.fence syncscope("workgroup") acquire{{$}}
  gpu.barrier memfence [#gpu.address_space<global>, #gpu.address_space<workgroup>]
  func.return
}

// CHECK-LABEL: func @barrier_private_only()
func.func @barrier_private_only() {
  // GFX9-NEXT: rocdl.s.barrier
  // GFX12-NEXT: rocdl.s.barrier.signal id = -1
  // GFX12-NEXT: rocdl.s.barrier.wait id = -1
  // CHECK-NOT: llvm.fence
  gpu.barrier memfence [#gpu.address_space<private>]
  func.return
}
}
