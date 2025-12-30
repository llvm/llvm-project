// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: @nvvm_tcgen05_cp_128x256b
llvm.func @nvvm_tcgen05_cp_128x256b(%taddr : !llvm.ptr<6>, %smem_desc : i64) {
  // CHECK: call void @llvm.nvvm.tcgen05.cp.128x256b.cg1(ptr addrspace(6) %{{.*}}, i64 %{{.*}})
  nvvm.tcgen05.cp %taddr, %smem_desc {shape = #nvvm.tcgen05_cp_shape<shape_128x256b>}

  // CHECK: call void @llvm.nvvm.tcgen05.cp.128x256b.cg2(ptr addrspace(6) %{{.*}}, i64 %{{.*}})
  nvvm.tcgen05.cp %taddr, %smem_desc {shape = #nvvm.tcgen05_cp_shape<shape_128x256b>, group = #nvvm.cta_group<cta_2>}

  // CHECK: call void @llvm.nvvm.tcgen05.cp.128x256b.b4x16_p64.cg2(ptr addrspace(6) %{{.*}}, i64 %{{.*}})
  nvvm.tcgen05.cp %taddr, %smem_desc {
    shape = #nvvm.tcgen05_cp_shape<shape_128x256b>,
    group = #nvvm.cta_group<cta_2>,
    srcFormat = #nvvm.tcgen05_cp_src_fmt<b4x16_p64>
  }
  // CHECK: call void @llvm.nvvm.tcgen05.cp.128x256b.b6x16_p32.cg2(ptr addrspace(6) %{{.*}}, i64 %{{.*}})
  nvvm.tcgen05.cp %taddr, %smem_desc {
    shape = #nvvm.tcgen05_cp_shape<shape_128x256b>,
    group = #nvvm.cta_group<cta_2>,
    srcFormat = #nvvm.tcgen05_cp_src_fmt<b6x16_p32>
  }
  llvm.return
}

// CHECK-LABEL: @nvvm_tcgen05_cp_4x256b
llvm.func @nvvm_tcgen05_cp_4x256b(%taddr : !llvm.ptr<6>, %smem_desc : i64) {
  // CHECK: call void @llvm.nvvm.tcgen05.cp.4x256b.cg1(ptr addrspace(6) %{{.*}}, i64 %{{.*}})
  nvvm.tcgen05.cp %taddr, %smem_desc {shape = #nvvm.tcgen05_cp_shape<shape_4x256b>}

  // CHECK: call void @llvm.nvvm.tcgen05.cp.4x256b.cg2(ptr addrspace(6) %{{.*}}, i64 %{{.*}})
  nvvm.tcgen05.cp %taddr, %smem_desc {shape = #nvvm.tcgen05_cp_shape<shape_4x256b>, group = #nvvm.cta_group<cta_2>}

  // CHECK: call void @llvm.nvvm.tcgen05.cp.4x256b.b4x16_p64.cg2(ptr addrspace(6) %{{.*}}, i64 %{{.*}})
  nvvm.tcgen05.cp %taddr, %smem_desc {
    shape = #nvvm.tcgen05_cp_shape<shape_4x256b>,
    group = #nvvm.cta_group<cta_2>,
    srcFormat = #nvvm.tcgen05_cp_src_fmt<b4x16_p64>
  }
  // CHECK: call void @llvm.nvvm.tcgen05.cp.4x256b.b6x16_p32.cg2(ptr addrspace(6) %{{.*}}, i64 %{{.*}})
  nvvm.tcgen05.cp %taddr, %smem_desc {
    shape = #nvvm.tcgen05_cp_shape<shape_4x256b>,
    group = #nvvm.cta_group<cta_2>,
    srcFormat = #nvvm.tcgen05_cp_src_fmt<b6x16_p32>
  }
  llvm.return
}

// CHECK-LABEL: @nvvm_tcgen05_cp_128x128b
llvm.func @nvvm_tcgen05_cp_128x128b(%taddr : !llvm.ptr<6>, %smem_desc : i64) {
  // CHECK: call void @llvm.nvvm.tcgen05.cp.128x128b.cg1(ptr addrspace(6) %{{.*}}, i64 %{{.*}})
  nvvm.tcgen05.cp %taddr, %smem_desc {shape = #nvvm.tcgen05_cp_shape<shape_128x128b>}

  // CHECK: call void @llvm.nvvm.tcgen05.cp.128x128b.cg2(ptr addrspace(6) %{{.*}}, i64 %{{.*}})
  nvvm.tcgen05.cp %taddr, %smem_desc {shape = #nvvm.tcgen05_cp_shape<shape_128x128b>, group = #nvvm.cta_group<cta_2>}

  // CHECK: call void @llvm.nvvm.tcgen05.cp.128x128b.b4x16_p64.cg2(ptr addrspace(6) %{{.*}}, i64 %{{.*}})
  nvvm.tcgen05.cp %taddr, %smem_desc {
    shape = #nvvm.tcgen05_cp_shape<shape_128x128b>,
    group = #nvvm.cta_group<cta_2>,
    srcFormat = #nvvm.tcgen05_cp_src_fmt<b4x16_p64>
  }
  // CHECK: call void @llvm.nvvm.tcgen05.cp.128x128b.b6x16_p32.cg2(ptr addrspace(6) %{{.*}}, i64 %{{.*}})
  nvvm.tcgen05.cp %taddr, %smem_desc {
    shape = #nvvm.tcgen05_cp_shape<shape_128x128b>,
    group = #nvvm.cta_group<cta_2>,
    srcFormat = #nvvm.tcgen05_cp_src_fmt<b6x16_p32>
  }
  llvm.return
}

// CHECK-LABEL: @nvvm_tcgen05_cp_64x128b
llvm.func @nvvm_tcgen05_cp_64x128b(%taddr : !llvm.ptr<6>, %smem_desc : i64) {
  // CHECK: call void @llvm.nvvm.tcgen05.cp.64x128b_warpx2_02_13.cg1(ptr addrspace(6) %{{.*}}, i64 %{{.*}})
  nvvm.tcgen05.cp %taddr, %smem_desc {
    shape = #nvvm.tcgen05_cp_shape<shape_64x128b>,
    multicast = #nvvm.tcgen05_cp_multicast<warpx2_02_13>
  }

  // CHECK: call void @llvm.nvvm.tcgen05.cp.64x128b_warpx2_02_13.cg2(ptr addrspace(6) %{{.*}}, i64 %{{.*}})
  nvvm.tcgen05.cp %taddr, %smem_desc {
    shape = #nvvm.tcgen05_cp_shape<shape_64x128b>,
    group = #nvvm.cta_group<cta_2>,
    multicast = #nvvm.tcgen05_cp_multicast<warpx2_02_13>
  }

  // CHECK: call void @llvm.nvvm.tcgen05.cp.64x128b_warpx2_02_13.b4x16_p64.cg1(ptr addrspace(6) %{{.*}}, i64 %{{.*}})
  nvvm.tcgen05.cp %taddr, %smem_desc {
    shape = #nvvm.tcgen05_cp_shape<shape_64x128b>,
    group = #nvvm.cta_group<cta_1>,
    multicast = #nvvm.tcgen05_cp_multicast<warpx2_02_13>,
    srcFormat = #nvvm.tcgen05_cp_src_fmt<b4x16_p64>
  }
  // CHECK: call void @llvm.nvvm.tcgen05.cp.64x128b_warpx2_01_23.b6x16_p32.cg2(ptr addrspace(6) %{{.*}}, i64 %{{.*}})
  nvvm.tcgen05.cp %taddr, %smem_desc {
    shape = #nvvm.tcgen05_cp_shape<shape_64x128b>,
    group = #nvvm.cta_group<cta_2>,
    multicast = #nvvm.tcgen05_cp_multicast<warpx2_01_23>,
    srcFormat = #nvvm.tcgen05_cp_src_fmt<b6x16_p32>
  }

  llvm.return
}

// CHECK-LABEL: @nvvm_tcgen05_cp_32x128b
llvm.func @nvvm_tcgen05_cp_32x128b(%taddr : !llvm.ptr<6>, %smem_desc : i64) {
  // CHECK: call void @llvm.nvvm.tcgen05.cp.32x128b_warpx4.cg1(ptr addrspace(6) %{{.*}}, i64 %{{.*}})
  nvvm.tcgen05.cp %taddr, %smem_desc {
    shape = #nvvm.tcgen05_cp_shape<shape_32x128b>,
    multicast = #nvvm.tcgen05_cp_multicast<warpx4>
  }

  // CHECK: call void @llvm.nvvm.tcgen05.cp.32x128b_warpx4.cg2(ptr addrspace(6) %{{.*}}, i64 %{{.*}})
  nvvm.tcgen05.cp %taddr, %smem_desc {
    shape = #nvvm.tcgen05_cp_shape<shape_32x128b>,
    group = #nvvm.cta_group<cta_2>,
    multicast = #nvvm.tcgen05_cp_multicast<warpx4>
  }

  // CHECK: call void @llvm.nvvm.tcgen05.cp.32x128b_warpx4.b4x16_p64.cg2(ptr addrspace(6) %{{.*}}, i64 %{{.*}})
  nvvm.tcgen05.cp %taddr, %smem_desc {
    shape = #nvvm.tcgen05_cp_shape<shape_32x128b>,
    group = #nvvm.cta_group<cta_2>,
    multicast = #nvvm.tcgen05_cp_multicast<warpx4>,
    srcFormat = #nvvm.tcgen05_cp_src_fmt<b4x16_p64>
  }
  // CHECK: call void @llvm.nvvm.tcgen05.cp.32x128b_warpx4.b6x16_p32.cg1(ptr addrspace(6) %{{.*}}, i64 %{{.*}})
  nvvm.tcgen05.cp %taddr, %smem_desc {
    shape = #nvvm.tcgen05_cp_shape<shape_32x128b>,
    group = #nvvm.cta_group<cta_1>,
    multicast = #nvvm.tcgen05_cp_multicast<warpx4>,
    srcFormat = #nvvm.tcgen05_cp_src_fmt<b6x16_p32>
  }

  llvm.return
}
