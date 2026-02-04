// RUN: mlir-translate --mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: @nvvm_tcgen05_ld_32x32b_min
llvm.func @nvvm_tcgen05_ld_32x32b_min(%addr : !llvm.ptr<6>) {

  // CHECK: {{.*}} = call { <2 x i32>, i32 } @llvm.nvvm.tcgen05.ld.red.32x32b.x2.i32(ptr addrspace(6) %{{.*}}, i32 0)
  // CHECK: {{.*}} = extractvalue { <2 x i32>, i32 } %{{.*}} 0
  // CHECK: {{.*}} = extractvalue { <2 x i32>, i32 } %{{.*}} 1
  // CHECK: {{.*}} = call { <4 x i32>, i32 } @llvm.nvvm.tcgen05.ld.red.32x32b.x4.i32(ptr addrspace(6) %{{.*}}, i32 0)
  // CHECK: {{.*}} = extractvalue { <4 x i32>, i32 } %{{.*}} 0
  // CHECK: {{.*}} = extractvalue { <4 x i32>, i32 } %{{.*}} 1
  // CHECK: {{.*}} = call { <8 x i32>, i32 } @llvm.nvvm.tcgen05.ld.red.32x32b.x8.i32(ptr addrspace(6) %{{.*}}, i32 0)
  // CHECK: {{.*}} = extractvalue { <8 x i32>, i32 } %{{.*}} 0
  // CHECK: %{{.*}} = extractvalue { <8 x i32>, i32 } %{{.*}} 1
  // CHECK: %{{.*}} = call { <16 x i32>, i32 } @llvm.nvvm.tcgen05.ld.red.32x32b.x16.i32(ptr addrspace(6) %{{.*}}, i32 0)
  // CHECK: %{{.*}} = extractvalue { <16 x i32>, i32 } %{{.*}}, 0
  // CHECK: %{{.*}} = extractvalue { <16 x i32>, i32 } %{{.*}}, 1
  // CHECK: %{{.*}} = call { <32 x i32>, i32 } @llvm.nvvm.tcgen05.ld.red.32x32b.x32.i32(ptr addrspace(6) %{{.*}}, i32 0)
  // CHECK: %{{.*}} = extractvalue { <32 x i32>, i32 } %{{.*}}, 0
  // CHECK: %{{.*}} = extractvalue { <32 x i32>, i32 } %{{.*}}, 1
  // CHECK: %{{.*}} = call { <64 x i32>, i32 } @llvm.nvvm.tcgen05.ld.red.32x32b.x64.i32(ptr addrspace(6) %{{.*}}, i32 0)
  // CHECK: %{{.*}} = extractvalue { <64 x i32>, i32 } %{{.*}}, 0
  // CHECK: %{{.*}} = extractvalue { <64 x i32>, i32 } %{{.*}}, 1
  // CHECK: %{{.*}} = call { <128 x i32>, i32 } @llvm.nvvm.tcgen05.ld.red.32x32b.x128.i32(ptr addrspace(6) %{{.*}}, i32 0)
  // CHECK: %{{.*}} = extractvalue { <128 x i32>, i32 } %{{.*}}, 0
  // CHECK: %{{.*}} = extractvalue { <128 x i32>, i32 } %{{.*}}, 1
  // CHECK: %{{.*}} = call { <2 x float>, float } @llvm.nvvm.tcgen05.ld.red.32x32b.x2.f32(ptr addrspace(6) %{{.*}}, i32 0, i1 false, i1 false)
  // CHECK: %{{.*}} = extractvalue { <2 x float>, float } %{{.*}}, 0
  // CHECK: %{{.*}} = extractvalue { <2 x float>, float } %{{.*}}, 1
  // CHECK: %{{.*}} = call { <4 x float>, float } @llvm.nvvm.tcgen05.ld.red.32x32b.x4.f32(ptr addrspace(6) %{{.*}}, i32 0, i1 false, i1 false)
  // CHECK: %{{.*}} = extractvalue { <4 x float>, float } %{{.*}}, 0
  // CHECK: %{{.*}} = extractvalue { <4 x float>, float } %{{.*}}, 1
  // CHECK: %{{.*}} = call { <8 x float>, float } @llvm.nvvm.tcgen05.ld.red.32x32b.x8.f32(ptr addrspace(6) %{{.*}}, i32 0, i1 false, i1 false)
  // CHECK: %{{.*}} = extractvalue { <8 x float>, float } %{{.*}}, 0
  // CHECK: %{{.*}} = extractvalue { <8 x float>, float } %{{.*}}, 1
  // CHECK: %{{.*}} = call { <16 x float>, float } @llvm.nvvm.tcgen05.ld.red.32x32b.x16.f32(ptr addrspace(6) %{{.*}}, i32 0, i1 false, i1 false)
  // CHECK: %{{.*}} = extractvalue { <16 x float>, float } %{{.*}}, 0
  // CHECK: %{{.*}} = extractvalue { <16 x float>, float } %{{.*}}, 1
  // CHECK: %{{.*}} = call { <32 x float>, float } @llvm.nvvm.tcgen05.ld.red.32x32b.x32.f32(ptr addrspace(6) %{{.*}}, i32 0, i1 false, i1 false)
  // CHECK: %{{.*}} = extractvalue { <32 x float>, float } %{{.*}}, 0
  // CHECK: %{{.*}} = extractvalue { <32 x float>, float } %{{.*}}, 1
  // CHECK: %{{.*}} = call { <64 x float>, float } @llvm.nvvm.tcgen05.ld.red.32x32b.x64.f32(ptr addrspace(6) %{{.*}}, i32 0, i1 false, i1 false)
  // CHECK: %{{.*}} = extractvalue { <64 x float>, float } %{{.*}}, 0
  // CHECK: %{{.*}} = extractvalue { <64 x float>, float } %{{.*}}, 1
  // CHECK: %{{.*}} = call { <128 x float>, float } @llvm.nvvm.tcgen05.ld.red.32x32b.x128.f32(ptr addrspace(6) %{{.*}}, i32 0, i1 false, i1 false)
  // CHECK: %{{.*}} = extractvalue { <128 x float>, float } %{{.*}}, 0
  // CHECK: %{{.*}} = extractvalue { <128 x float>, float } %{{.*}}, 1

  %data, %redval = nvvm.tcgen05.ld.red min %addr { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>} : vector<2 x i32>, i32

  %data1, %redval1 = nvvm.tcgen05.ld.red min %addr { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>} : vector<4 x i32>, i32

  %data2, %redval2 = nvvm.tcgen05.ld.red min %addr { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>} : vector<8 x i32>, i32

  %data3, %redval3 = nvvm.tcgen05.ld.red min %addr { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>} : vector<16 x i32>, i32

  %data4, %redval4 = nvvm.tcgen05.ld.red min %addr { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>} : vector<32 x i32>, i32

  %data5, %redval5 = nvvm.tcgen05.ld.red min %addr { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>} : vector<64 x i32>, i32

  %data6, %redval6 = nvvm.tcgen05.ld.red min %addr { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>} : vector<128 x i32>, i32

  %data7, %redval7 = nvvm.tcgen05.ld.red min %addr { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>} : vector<2 x f32>, f32

  %data8, %redval8 = nvvm.tcgen05.ld.red min %addr { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>} : vector<4 x f32>, f32

  %data9, %redval9 = nvvm.tcgen05.ld.red min %addr { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>} : vector<8 x f32>, f32

  %data10, %redval10 = nvvm.tcgen05.ld.red min %addr { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>} : vector<16 x f32>, f32

  %data11, %redval11 = nvvm.tcgen05.ld.red min %addr { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>} : vector<32 x f32>, f32

  %data12, %redval12 = nvvm.tcgen05.ld.red min %addr { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>} : vector<64 x f32>, f32

  %data13, %redval13 = nvvm.tcgen05.ld.red min %addr { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>} : vector<128 x f32>, f32
  llvm.return
}

// CHECK-LABEL: @nvvm_tcgen05_ld_32x32b_max
llvm.func @nvvm_tcgen05_ld_32x32b_max(%addr : !llvm.ptr<6>) {

  // CHECK: {{.*}} = call { <2 x i32>, i32 } @llvm.nvvm.tcgen05.ld.red.32x32b.x2.i32(ptr addrspace(6) %{{.*}}, i32 1)
  // CHECK: {{.*}} = extractvalue { <2 x i32>, i32 } %{{.*}} 0
  // CHECK: {{.*}} = extractvalue { <2 x i32>, i32 } %{{.*}} 1
  // CHECK: {{.*}} = call { <4 x i32>, i32 } @llvm.nvvm.tcgen05.ld.red.32x32b.x4.i32(ptr addrspace(6) %{{.*}}, i32 1)
  // CHECK: {{.*}} = extractvalue { <4 x i32>, i32 } %{{.*}} 0
  // CHECK: {{.*}} = extractvalue { <4 x i32>, i32 } %{{.*}} 1
  // CHECK: {{.*}} = call { <8 x i32>, i32 } @llvm.nvvm.tcgen05.ld.red.32x32b.x8.i32(ptr addrspace(6) %{{.*}}, i32 1)
  // CHECK: {{.*}} = extractvalue { <8 x i32>, i32 } %{{.*}} 0
  // CHECK: %{{.*}} = extractvalue { <8 x i32>, i32 } %{{.*}} 1
  // CHECK: %{{.*}} = call { <16 x i32>, i32 } @llvm.nvvm.tcgen05.ld.red.32x32b.x16.i32(ptr addrspace(6) %{{.*}}, i32 1)
  // CHECK: %{{.*}} = extractvalue { <16 x i32>, i32 } %{{.*}}, 0
  // CHECK: %{{.*}} = extractvalue { <16 x i32>, i32 } %{{.*}}, 1
  // CHECK: %{{.*}} = call { <32 x i32>, i32 } @llvm.nvvm.tcgen05.ld.red.32x32b.x32.i32(ptr addrspace(6) %{{.*}}, i32 1)
  // CHECK: %{{.*}} = extractvalue { <32 x i32>, i32 } %{{.*}}, 0
  // CHECK: %{{.*}} = extractvalue { <32 x i32>, i32 } %{{.*}}, 1
  // CHECK: %{{.*}} = call { <64 x i32>, i32 } @llvm.nvvm.tcgen05.ld.red.32x32b.x64.i32(ptr addrspace(6) %{{.*}}, i32 1)
  // CHECK: %{{.*}} = extractvalue { <64 x i32>, i32 } %{{.*}}, 0
  // CHECK: %{{.*}} = extractvalue { <64 x i32>, i32 } %{{.*}}, 1
  // CHECK: %{{.*}} = call { <128 x i32>, i32 } @llvm.nvvm.tcgen05.ld.red.32x32b.x128.i32(ptr addrspace(6) %{{.*}}, i32 1)
  // CHECK: %{{.*}} = extractvalue { <128 x i32>, i32 } %{{.*}}, 0
  // CHECK: %{{.*}} = extractvalue { <128 x i32>, i32 } %{{.*}}, 1
  // CHECK: %{{.*}} = call { <2 x float>, float } @llvm.nvvm.tcgen05.ld.red.32x32b.x2.f32(ptr addrspace(6) %{{.*}}, i32 1, i1 false, i1 false)
  // CHECK: %{{.*}} = extractvalue { <2 x float>, float } %{{.*}}, 0
  // CHECK: %{{.*}} = extractvalue { <2 x float>, float } %{{.*}}, 1
  // CHECK: %{{.*}} = call { <4 x float>, float } @llvm.nvvm.tcgen05.ld.red.32x32b.x4.f32(ptr addrspace(6) %{{.*}}, i32 1, i1 false, i1 false)
  // CHECK: %{{.*}} = extractvalue { <4 x float>, float } %{{.*}}, 0
  // CHECK: %{{.*}} = extractvalue { <4 x float>, float } %{{.*}}, 1
  // CHECK: %{{.*}} = call { <8 x float>, float } @llvm.nvvm.tcgen05.ld.red.32x32b.x8.f32(ptr addrspace(6) %{{.*}}, i32 1, i1 false, i1 false)
  // CHECK: %{{.*}} = extractvalue { <8 x float>, float } %{{.*}}, 0
  // CHECK: %{{.*}} = extractvalue { <8 x float>, float } %{{.*}}, 1
  // CHECK: %{{.*}} = call { <16 x float>, float } @llvm.nvvm.tcgen05.ld.red.32x32b.x16.f32(ptr addrspace(6) %{{.*}}, i32 1, i1 false, i1 false)
  // CHECK: %{{.*}} = extractvalue { <16 x float>, float } %{{.*}}, 0
  // CHECK: %{{.*}} = extractvalue { <16 x float>, float } %{{.*}}, 1
  // CHECK: %{{.*}} = call { <32 x float>, float } @llvm.nvvm.tcgen05.ld.red.32x32b.x32.f32(ptr addrspace(6) %{{.*}}, i32 1, i1 false, i1 false)
  // CHECK: %{{.*}} = extractvalue { <32 x float>, float } %{{.*}}, 0
  // CHECK: %{{.*}} = extractvalue { <32 x float>, float } %{{.*}}, 1
  // CHECK: %{{.*}} = call { <64 x float>, float } @llvm.nvvm.tcgen05.ld.red.32x32b.x64.f32(ptr addrspace(6) %{{.*}}, i32 1, i1 false, i1 false)
  // CHECK: %{{.*}} = extractvalue { <64 x float>, float } %{{.*}}, 0
  // CHECK: %{{.*}} = extractvalue { <64 x float>, float } %{{.*}}, 1
  // CHECK: %{{.*}} = call { <128 x float>, float } @llvm.nvvm.tcgen05.ld.red.32x32b.x128.f32(ptr addrspace(6) %{{.*}}, i32 1, i1 false, i1 false)
  // CHECK: %{{.*}} = extractvalue { <128 x float>, float } %{{.*}}, 0
  // CHECK: %{{.*}} = extractvalue { <128 x float>, float } %{{.*}}, 1

  %data, %redval = nvvm.tcgen05.ld.red max %addr { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>} : vector<2 x i32>, i32

  %data1, %redval1 = nvvm.tcgen05.ld.red max %addr { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>} : vector<4 x i32>, i32

  %data2, %redval2 = nvvm.tcgen05.ld.red max %addr { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>} : vector<8 x i32>, i32

  %data3, %redval3 = nvvm.tcgen05.ld.red max %addr { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>} : vector<16 x i32>, i32

  %data4, %redval4 = nvvm.tcgen05.ld.red max %addr { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>} : vector<32 x i32>, i32

  %data5, %redval5 = nvvm.tcgen05.ld.red max %addr { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>} : vector<64 x i32>, i32

  %data6, %redval6 = nvvm.tcgen05.ld.red max %addr { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>} : vector<128 x i32>, i32

  %data7, %redval7 = nvvm.tcgen05.ld.red max %addr { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>} : vector<2 x f32>, f32

  %data8, %redval8 = nvvm.tcgen05.ld.red max %addr { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>} : vector<4 x f32>, f32

  %data9, %redval9 = nvvm.tcgen05.ld.red max %addr { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>} : vector<8 x f32>, f32

  %data10, %redval10 = nvvm.tcgen05.ld.red max %addr { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>} : vector<16 x f32>, f32

  %data11, %redval11 = nvvm.tcgen05.ld.red max %addr { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>} : vector<32 x f32>, f32

  %data12, %redval12 = nvvm.tcgen05.ld.red max %addr { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>} : vector<64 x f32>, f32

  %data13, %redval13 = nvvm.tcgen05.ld.red max %addr { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>} : vector<128 x f32>, f32
  llvm.return
}

// CHECK-LABEL: @nvvm_tcgen05_ld_32x32b_min_abs_nan
llvm.func @nvvm_tcgen05_ld_32x32b_min_abs_nan(%addr : !llvm.ptr<6>) {

  // CHECK: %{{.*}} = call { <2 x float>, float } @llvm.nvvm.tcgen05.ld.red.32x32b.x2.f32(ptr addrspace(6) %{{.*}}, i32 0, i1 true, i1 true)
  // CHECK: %{{.*}} = extractvalue { <2 x float>, float } %{{.*}}, 0
  // CHECK: %{{.*}} = extractvalue { <2 x float>, float } %{{.*}}, 1
  // CHECK: %{{.*}} = call { <4 x float>, float } @llvm.nvvm.tcgen05.ld.red.32x32b.x4.f32(ptr addrspace(6) %{{.*}}, i32 0, i1 true, i1 true)
  // CHECK: %{{.*}} = extractvalue { <4 x float>, float } %{{.*}}, 0
  // CHECK: %{{.*}} = extractvalue { <4 x float>, float } %{{.*}}, 1
  // CHECK: %{{.*}} = call { <8 x float>, float } @llvm.nvvm.tcgen05.ld.red.32x32b.x8.f32(ptr addrspace(6) %{{.*}}, i32 0, i1 true, i1 true)
  // CHECK: %{{.*}} = extractvalue { <8 x float>, float } %{{.*}}, 0
  // CHECK: %{{.*}} = extractvalue { <8 x float>, float } %{{.*}}, 1
  // CHECK: %{{.*}} = call { <16 x float>, float } @llvm.nvvm.tcgen05.ld.red.32x32b.x16.f32(ptr addrspace(6) %{{.*}}, i32 0, i1 true, i1 true)
  // CHECK: %{{.*}} = extractvalue { <16 x float>, float } %{{.*}}, 0
  // CHECK: %{{.*}} = extractvalue { <16 x float>, float } %{{.*}}, 1
  // CHECK: %{{.*}} = call { <32 x float>, float } @llvm.nvvm.tcgen05.ld.red.32x32b.x32.f32(ptr addrspace(6) %{{.*}}, i32 0, i1 true, i1 true)
  // CHECK: %{{.*}} = extractvalue { <32 x float>, float } %{{.*}}, 0
  // CHECK: %{{.*}} = extractvalue { <32 x float>, float } %{{.*}}, 1
  // CHECK: %{{.*}} = call { <64 x float>, float } @llvm.nvvm.tcgen05.ld.red.32x32b.x64.f32(ptr addrspace(6) %{{.*}}, i32 0, i1 true, i1 true)
  // CHECK: %{{.*}} = extractvalue { <64 x float>, float } %{{.*}}, 0
  // CHECK: %{{.*}} = extractvalue { <64 x float>, float } %{{.*}}, 1
  // CHECK: %{{.*}} = call { <128 x float>, float } @llvm.nvvm.tcgen05.ld.red.32x32b.x128.f32(ptr addrspace(6) %{{.*}}, i32 0, i1 true, i1 true)
  // CHECK: %{{.*}} = extractvalue { <128 x float>, float } %{{.*}}, 0
  // CHECK: %{{.*}} = extractvalue { <128 x float>, float } %{{.*}}, 1

  %data7, %redval7 = nvvm.tcgen05.ld.red min %addr { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>, abs, nan} : vector<2 x f32>, f32

  %data8, %redval8 = nvvm.tcgen05.ld.red min %addr { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>, abs, nan} : vector<4 x f32>, f32

  %data9, %redval9 = nvvm.tcgen05.ld.red min %addr { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>, abs, nan} : vector<8 x f32>, f32

  %data10, %redval10 = nvvm.tcgen05.ld.red min %addr { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>, abs, nan} : vector<16 x f32>, f32

  %data11, %redval11 = nvvm.tcgen05.ld.red min %addr { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>, abs, nan} : vector<32 x f32>, f32

  %data12, %redval12 = nvvm.tcgen05.ld.red min %addr { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>, abs, nan} : vector<64 x f32>, f32

  %data13, %redval13 = nvvm.tcgen05.ld.red min %addr { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>, abs, nan} : vector<128 x f32>, f32
  llvm.return
}

// CHECK-LABEL: @nvvm_tcgen05_ld_32x32b_max_abs_nan
llvm.func @nvvm_tcgen05_ld_32x32b_max_abs_nan(%addr : !llvm.ptr<6>) {

  // CHECK: %{{.*}} = call { <2 x float>, float } @llvm.nvvm.tcgen05.ld.red.32x32b.x2.f32(ptr addrspace(6) %{{.*}}, i32 1, i1 true, i1 true)
  // CHECK: %{{.*}} = extractvalue { <2 x float>, float } %{{.*}}, 0
  // CHECK: %{{.*}} = extractvalue { <2 x float>, float } %{{.*}}, 1
  // CHECK: %{{.*}} = call { <4 x float>, float } @llvm.nvvm.tcgen05.ld.red.32x32b.x4.f32(ptr addrspace(6) %{{.*}}, i32 1, i1 true, i1 true)
  // CHECK: %{{.*}} = extractvalue { <4 x float>, float } %{{.*}}, 0
  // CHECK: %{{.*}} = extractvalue { <4 x float>, float } %{{.*}}, 1
  // CHECK: %{{.*}} = call { <8 x float>, float } @llvm.nvvm.tcgen05.ld.red.32x32b.x8.f32(ptr addrspace(6) %{{.*}}, i32 1, i1 true, i1 true)
  // CHECK: %{{.*}} = extractvalue { <8 x float>, float } %{{.*}}, 0
  // CHECK: %{{.*}} = extractvalue { <8 x float>, float } %{{.*}}, 1
  // CHECK: %{{.*}} = call { <16 x float>, float } @llvm.nvvm.tcgen05.ld.red.32x32b.x16.f32(ptr addrspace(6) %{{.*}}, i32 1, i1 true, i1 true)
  // CHECK: %{{.*}} = extractvalue { <16 x float>, float } %{{.*}}, 0
  // CHECK: %{{.*}} = extractvalue { <16 x float>, float } %{{.*}}, 1
  // CHECK: %{{.*}} = call { <32 x float>, float } @llvm.nvvm.tcgen05.ld.red.32x32b.x32.f32(ptr addrspace(6) %{{.*}}, i32 1, i1 true, i1 true)
  // CHECK: %{{.*}} = extractvalue { <32 x float>, float } %{{.*}}, 0
  // CHECK: %{{.*}} = extractvalue { <32 x float>, float } %{{.*}}, 1
  // CHECK: %{{.*}} = call { <64 x float>, float } @llvm.nvvm.tcgen05.ld.red.32x32b.x64.f32(ptr addrspace(6) %{{.*}}, i32 1, i1 true, i1 true)
  // CHECK: %{{.*}} = extractvalue { <64 x float>, float } %{{.*}}, 0
  // CHECK: %{{.*}} = extractvalue { <64 x float>, float } %{{.*}}, 1
  // CHECK: %{{.*}} = call { <128 x float>, float } @llvm.nvvm.tcgen05.ld.red.32x32b.x128.f32(ptr addrspace(6) %{{.*}}, i32 1, i1 true, i1 true)
  // CHECK: %{{.*}} = extractvalue { <128 x float>, float } %{{.*}}, 0
  // CHECK: %{{.*}} = extractvalue { <128 x float>, float } %{{.*}}, 1

  %data7, %redval7 = nvvm.tcgen05.ld.red max %addr { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>, abs, nan} : vector<2 x f32>, f32

  %data8, %redval8 = nvvm.tcgen05.ld.red max %addr { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>, abs, nan} : vector<4 x f32>, f32

  %data9, %redval9 = nvvm.tcgen05.ld.red max %addr { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>, abs, nan} : vector<8 x f32>, f32

  %data10, %redval10 = nvvm.tcgen05.ld.red max %addr { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>, abs, nan} : vector<16 x f32>, f32

  %data11, %redval11 = nvvm.tcgen05.ld.red max %addr { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>, abs, nan} : vector<32 x f32>, f32

  %data12, %redval12 = nvvm.tcgen05.ld.red max %addr { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>, abs, nan} : vector<64 x f32>, f32

  %data13, %redval13 = nvvm.tcgen05.ld.red max %addr { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>, abs, nan} : vector<128 x f32>, f32
  llvm.return
}

// CHECK-LABEL: @nvvm_tcgen05_ld_16x32bx2_min
llvm.func @nvvm_tcgen05_ld_16x32bx2_min(%addr : !llvm.ptr<6>) {

  %offset = llvm.mlir.constant(0: i64) : i64

  // CHECK %{{.*}} = call { <2 x i32>, i32 } @llvm.nvvm.tcgen05.ld.red.16x32bx2.x2.i32(ptr addrspace(6) %{{.*}}, i64 0, i32 0)
  // CHECK %{{.*}} = extractvalue { <2 x i32>, i32 } {{.*}}, 0
  // CHECK %{{.*}} = extractvalue { <2 x i32>, i32 } {{.*}}, 1
  // CHECK %{{.*}} = call { <4 x i32>, i32 } @llvm.nvvm.tcgen05.ld.red.16x32bx2.x4.i32(ptr addrspace(6) %{{.*}}, i64 0, i32 0)
  // CHECK %{{.*}} = extractvalue { <4 x i32>, i32 } {{.*}}, 0
  // CHECK %{{.*}} = extractvalue { <4 x i32>, i32 } {{.*}}, 1
  // CHECK %{{.*}} = call { <8 x i32>, i32 } @llvm.nvvm.tcgen05.ld.red.16x32bx2.x8.i32(ptr addrspace(6) %{{.*}}, i64 0, i32 0)
  // CHECK %{{.*}} = extractvalue { <8 x i32>, i32 } {{.*}}, 0
  // CHECK %{{.*}} = extractvalue { <8 x i32>, i32 } {{.*}}, 1
  // CHECK %{{.*}} = call { <16 x i32>, i32 } @llvm.nvvm.tcgen05.ld.red.16x32bx2.x16.i32(ptr addrspace(6) %{{.*}}, i64 0, i32 0)
  // CHECK %{{.*}} = extractvalue { <16 x i32>, i32 } %{{.*}}, 0
  // CHECK %{{.*}} = extractvalue { <16 x i32>, i32 } %{{.*}}, 1
  // CHECK %{{.*}} = call { <32 x i32>, i32 } @llvm.nvvm.tcgen05.ld.red.16x32bx2.x32.i32(ptr addrspace(6) %{{.*}}, i64 0, i32 0)
  // CHECK %{{.*}} = extractvalue { <32 x i32>, i32 } %{{.*}}, 0
  // CHECK %{{.*}} = extractvalue { <32 x i32>, i32 } %{{.*}}, 1
  // CHECK %{{.*}} = call { <64 x i32>, i32 } @llvm.nvvm.tcgen05.ld.red.16x32bx2.x64.i32(ptr addrspace(6) %{{.*}}, i64 0, i32 0)
  // CHECK %{{.*}} = extractvalue { <64 x i32>, i32 } %{{.*}}, 0
  // CHECK %{{.*}} = extractvalue { <64 x i32>, i32 } %{{.*}}, 1
  // CHECK %{{.*}} = call { <128 x i32>, i32 } @llvm.nvvm.tcgen05.ld.red.16x32bx2.x128.i32(ptr addrspace(6) %{{.*}}, i64 0, i32 0)
  // CHECK %{{.*}} = extractvalue { <128 x i32>, i32 } %{{.*}}, 0
  // CHECK %{{.*}} = extractvalue { <128 x i32>, i32 } %{{.*}}, 1
  // CHECK %{{.*}} = call { <2 x float>, float } @llvm.nvvm.tcgen05.ld.red.16x32bx2.x2.f32(ptr addrspace(6) %{{.*}}, i64 0, i32 0, i1 false, i1 false)
  // CHECK %{{.*}} = extractvalue { <2 x float>, float } %{{.*}}, 0
  // CHECK %{{.*}} = extractvalue { <2 x float>, float } %{{.*}}, 1
  // CHECK %{{.*}} = call { <4 x float>, float } @llvm.nvvm.tcgen05.ld.red.16x32bx2.x4.f32(ptr addrspace(6) %{{.*}}, i64 0, i32 0, i1 false, i1 false)
  // CHECK %{{.*}} = extractvalue { <4 x float>, float } %{{.*}}, 0
  // CHECK %{{.*}} = extractvalue { <4 x float>, float } %{{.*}}, 1
  // CHECK %{{.*}} = call { <8 x float>, float } @llvm.nvvm.tcgen05.ld.red.16x32bx2.x8.f32(ptr addrspace(6) %{{.*}}, i64 0, i32 0, i1 false, i1 false)
  // CHECK %{{.*}} = extractvalue { <8 x float>, float } %{{.*}}, 0
  // CHECK %{{.*}} = extractvalue { <8 x float>, float } %{{.*}}, 1
  // CHECK %{{.*}} = call { <16 x float>, float } @llvm.nvvm.tcgen05.ld.red.16x32bx2.x16.f32(ptr addrspace(6) %{{.*}}, i64 0, i32 0, i1 false, i1 false)
  // CHECK %{{.*}} = extractvalue { <16 x float>, float } %{{.*}}, 0
  // CHECK %{{.*}} = extractvalue { <16 x float>, float } %{{.*}}, 1
  // CHECK %{{.*}} = call { <32 x float>, float } @llvm.nvvm.tcgen05.ld.red.16x32bx2.x32.f32(ptr addrspace(6) %{{.*}}, i64 0, i32 0, i1 false, i1 false)
  // CHECK %{{.*}} = extractvalue { <32 x float>, float } %{{.*}}, 0
  // CHECK %{{.*}} = extractvalue { <32 x float>, float } %{{.*}}, 1
  // CHECK %{{.*}} = call { <64 x float>, float } @llvm.nvvm.tcgen05.ld.red.16x32bx2.x64.f32(ptr addrspace(6) %{{.*}}, i64 0, i32 0, i1 false, i1 false)
  // CHECK %{{.*}} = extractvalue { <64 x float>, float } %{{.*}}, 0
  // CHECK %{{.*}} = extractvalue { <64 x float>, float } %{{.*}}, 1
  // CHECK %{{.*}} = call { <128 x float>, float } @llvm.nvvm.tcgen05.ld.red.16x32bx2.x128.f32(ptr addrspace(6) %{{.*}}, i64 0, i32 0, i1 false, i1 false)
  // CHECK %{{.*}} = extractvalue { <128 x float>, float } %{{.*}}, 0
  // CHECK %{{.*}} = extractvalue { <128 x float>, float } %{{.*}}, 1

  %data, %redval = nvvm.tcgen05.ld.red min %addr, %offset { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>} : vector<2 x i32>, i32

  %data1, %redval1 = nvvm.tcgen05.ld.red min %addr, %offset { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>} : vector<4 x i32>, i32

  %data2, %redval2 = nvvm.tcgen05.ld.red min %addr, %offset { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>} : vector<8 x i32>, i32

  %data3, %redval3 = nvvm.tcgen05.ld.red min %addr, %offset { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>} : vector<16 x i32>, i32

  %data4, %redval4 = nvvm.tcgen05.ld.red min %addr, %offset { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>} : vector<32 x i32>, i32

  %data5, %redval5 = nvvm.tcgen05.ld.red min %addr, %offset { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>} : vector<64 x i32>, i32

  %data6, %redval6 = nvvm.tcgen05.ld.red min %addr, %offset { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>} : vector<128 x i32>, i32

  %data7, %redval7 = nvvm.tcgen05.ld.red min %addr, %offset { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>} : vector<2 x f32>, f32

  %data8, %redval8 = nvvm.tcgen05.ld.red min %addr, %offset { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>} : vector<4 x f32>, f32

  %data9, %redval9 = nvvm.tcgen05.ld.red min %addr, %offset { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>} : vector<8 x f32>, f32

  %data10, %redval10 = nvvm.tcgen05.ld.red min %addr, %offset { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>} : vector<16 x f32>, f32

  %data11, %redval11 = nvvm.tcgen05.ld.red min %addr, %offset { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>} : vector<32 x f32>, f32

  %data12, %redval12 = nvvm.tcgen05.ld.red min %addr, %offset { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>} : vector<64 x f32>, f32

  %data13, %redval13 = nvvm.tcgen05.ld.red min %addr, %offset { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>} : vector<128 x f32>, f32
  llvm.return
}

// CHECK-LABEL: @nvvm_tcgen05_ld_16x32bx2_max
llvm.func @nvvm_tcgen05_ld_16x32bx2_max(%addr : !llvm.ptr<6>) {

  %offset = llvm.mlir.constant(0: i64) : i64

  // CHECK %{{.*}} = call { <2 x i32>, i32 } @llvm.nvvm.tcgen05.ld.red.16x32bx2.x2.i32(ptr addrspace(6) %{{.*}}, i64 0, i32 1)
  // CHECK %{{.*}} = extractvalue { <2 x i32>, i32 } {{.*}}, 0
  // CHECK %{{.*}} = extractvalue { <2 x i32>, i32 } {{.*}}, 1
  // CHECK %{{.*}} = call { <4 x i32>, i32 } @llvm.nvvm.tcgen05.ld.red.16x32bx2.x4.i32(ptr addrspace(6) %{{.*}}, i64 0, i32 1)
  // CHECK %{{.*}} = extractvalue { <4 x i32>, i32 } {{.*}}, 0
  // CHECK %{{.*}} = extractvalue { <4 x i32>, i32 } {{.*}}, 1
  // CHECK %{{.*}} = call { <8 x i32>, i32 } @llvm.nvvm.tcgen05.ld.red.16x32bx2.x8.i32(ptr addrspace(6) %{{.*}}, i64 0, i32 1)
  // CHECK %{{.*}} = extractvalue { <8 x i32>, i32 } {{.*}}, 0
  // CHECK %{{.*}} = extractvalue { <8 x i32>, i32 } {{.*}}, 1
  // CHECK %{{.*}} = call { <16 x i32>, i32 } @llvm.nvvm.tcgen05.ld.red.16x32bx2.x16.i32(ptr addrspace(6) %{{.*}}, i64 0, i32 1)
  // CHECK %{{.*}} = extractvalue { <16 x i32>, i32 } %{{.*}}, 0
  // CHECK %{{.*}} = extractvalue { <16 x i32>, i32 } %{{.*}}, 1
  // CHECK %{{.*}} = call { <32 x i32>, i32 } @llvm.nvvm.tcgen05.ld.red.16x32bx2.x32.i32(ptr addrspace(6) %{{.*}}, i64 0, i32 1)
  // CHECK %{{.*}} = extractvalue { <32 x i32>, i32 } %{{.*}}, 0
  // CHECK %{{.*}} = extractvalue { <32 x i32>, i32 } %{{.*}}, 1
  // CHECK %{{.*}} = call { <64 x i32>, i32 } @llvm.nvvm.tcgen05.ld.red.16x32bx2.x64.i32(ptr addrspace(6) %{{.*}}, i64 0, i32 1)
  // CHECK %{{.*}} = extractvalue { <64 x i32>, i32 } %{{.*}}, 0
  // CHECK %{{.*}} = extractvalue { <64 x i32>, i32 } %{{.*}}, 1
  // CHECK %{{.*}} = call { <128 x i32>, i32 } @llvm.nvvm.tcgen05.ld.red.16x32bx2.x128.i32(ptr addrspace(6) %{{.*}}, i64 0, i32 1)
  // CHECK %{{.*}} = extractvalue { <128 x i32>, i32 } %{{.*}}, 0
  // CHECK %{{.*}} = extractvalue { <128 x i32>, i32 } %{{.*}}, 1
  // CHECK %{{.*}} = call { <2 x float>, float } @llvm.nvvm.tcgen05.ld.red.16x32bx2.x2.f32(ptr addrspace(6) %{{.*}}, i64 0, i32 1, i1 false, i1 false)
  // CHECK %{{.*}} = extractvalue { <2 x float>, float } %{{.*}}, 0
  // CHECK %{{.*}} = extractvalue { <2 x float>, float } %{{.*}}, 1
  // CHECK %{{.*}} = call { <4 x float>, float } @llvm.nvvm.tcgen05.ld.red.16x32bx2.x4.f32(ptr addrspace(6) %{{.*}}, i64 0, i32 1, i1 false, i1 false)
  // CHECK %{{.*}} = extractvalue { <4 x float>, float } %{{.*}}, 0
  // CHECK %{{.*}} = extractvalue { <4 x float>, float } %{{.*}}, 1
  // CHECK %{{.*}} = call { <8 x float>, float } @llvm.nvvm.tcgen05.ld.red.16x32bx2.x8.f32(ptr addrspace(6) %{{.*}}, i64 0, i32 1, i1 false, i1 false)
  // CHECK %{{.*}} = extractvalue { <8 x float>, float } %{{.*}}, 0
  // CHECK %{{.*}} = extractvalue { <8 x float>, float } %{{.*}}, 1
  // CHECK %{{.*}} = call { <16 x float>, float } @llvm.nvvm.tcgen05.ld.red.16x32bx2.x16.f32(ptr addrspace(6) %{{.*}}, i64 0, i32 1, i1 false, i1 false)
  // CHECK %{{.*}} = extractvalue { <16 x float>, float } %{{.*}}, 0
  // CHECK %{{.*}} = extractvalue { <16 x float>, float } %{{.*}}, 1
  // CHECK %{{.*}} = call { <32 x float>, float } @llvm.nvvm.tcgen05.ld.red.16x32bx2.x32.f32(ptr addrspace(6) %{{.*}}, i64 0, i32 1, i1 false, i1 false)
  // CHECK %{{.*}} = extractvalue { <32 x float>, float } %{{.*}}, 0
  // CHECK %{{.*}} = extractvalue { <32 x float>, float } %{{.*}}, 1
  // CHECK %{{.*}} = call { <64 x float>, float } @llvm.nvvm.tcgen05.ld.red.16x32bx2.x64.f32(ptr addrspace(6) %{{.*}}, i64 0, i32 1, i1 false, i1 false)
  // CHECK %{{.*}} = extractvalue { <64 x float>, float } %{{.*}}, 0
  // CHECK %{{.*}} = extractvalue { <64 x float>, float } %{{.*}}, 1
  // CHECK %{{.*}} = call { <128 x float>, float } @llvm.nvvm.tcgen05.ld.red.16x32bx2.x128.f32(ptr addrspace(6) %{{.*}}, i64 0, i32 1, i1 false, i1 false)
  // CHECK %{{.*}} = extractvalue { <128 x float>, float } %{{.*}}, 0
  // CHECK %{{.*}} = extractvalue { <128 x float>, float } %{{.*}}, 1

  %data, %redval = nvvm.tcgen05.ld.red max %addr, %offset { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>} : vector<2 x i32>, i32

  %data1, %redval1 = nvvm.tcgen05.ld.red max %addr, %offset { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>} : vector<4 x i32>, i32

  %data2, %redval2 = nvvm.tcgen05.ld.red max %addr, %offset { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>} : vector<8 x i32>, i32

  %data3, %redval3 = nvvm.tcgen05.ld.red max %addr, %offset { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>} : vector<16 x i32>, i32

  %data4, %redval4 = nvvm.tcgen05.ld.red max %addr, %offset { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>} : vector<32 x i32>, i32

  %data5, %redval5 = nvvm.tcgen05.ld.red max %addr, %offset { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>} : vector<64 x i32>, i32

  %data6, %redval6 = nvvm.tcgen05.ld.red max %addr, %offset { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>} : vector<128 x i32>, i32

  %data7, %redval7 = nvvm.tcgen05.ld.red max %addr, %offset { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>} : vector<2 x f32>, f32

  %data8, %redval8 = nvvm.tcgen05.ld.red max %addr, %offset { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>} : vector<4 x f32>, f32

  %data9, %redval9 = nvvm.tcgen05.ld.red max %addr, %offset { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>} : vector<8 x f32>, f32

  %data10, %redval10 = nvvm.tcgen05.ld.red max %addr, %offset { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>} : vector<16 x f32>, f32

  %data11, %redval11 = nvvm.tcgen05.ld.red max %addr, %offset { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>} : vector<32 x f32>, f32

  %data12, %redval12 = nvvm.tcgen05.ld.red max %addr, %offset { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>} : vector<64 x f32>, f32

  %data13, %redval13 = nvvm.tcgen05.ld.red max %addr, %offset { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>} : vector<128 x f32>, f32
  llvm.return
}

// CHECK-LABEL: @nvvm_tcgen05_ld_16x32bx2_min_nan_abs
llvm.func @nvvm_tcgen05_ld_16x32bx2_min_nan_abs(%addr : !llvm.ptr<6>) {

  %offset = llvm.mlir.constant(0: i64) : i64
  // CHECK %{{.*}} = call { <2 x float>, float } @llvm.nvvm.tcgen05.ld.red.16x32bx2.x2.f32(ptr addrspace(6) %{{.*}}, i64 0, i32 0, i1 true, i1 true)
  // CHECK %{{.*}} = extractvalue { <2 x float>, float } %{{.*}}, 0
  // CHECK %{{.*}} = extractvalue { <2 x float>, float } %{{.*}}, 1
  // CHECK %{{.*}} = call { <4 x float>, float } @llvm.nvvm.tcgen05.ld.red.16x32bx2.x4.f32(ptr addrspace(6) %{{.*}}, i64 0, i32 0, i1 true, i1 true)
  // CHECK %{{.*}} = extractvalue { <4 x float>, float } %{{.*}}, 0
  // CHECK %{{.*}} = extractvalue { <4 x float>, float } %{{.*}}, 1
  // CHECK %{{.*}} = call { <8 x float>, float } @llvm.nvvm.tcgen05.ld.red.16x32bx2.x8.f32(ptr addrspace(6) %{{.*}}, i64 0, i32 0, i1 true, i1 true)
  // CHECK %{{.*}} = extractvalue { <8 x float>, float } %{{.*}}, 0
  // CHECK %{{.*}} = extractvalue { <8 x float>, float } %{{.*}}, 1
  // CHECK %{{.*}} = call { <16 x float>, float } @llvm.nvvm.tcgen05.ld.red.16x32bx2.x16.f32(ptr addrspace(6) %{{.*}}, i64 0, i32 0, i1 true, i1 true)
  // CHECK %{{.*}} = extractvalue { <16 x float>, float } %{{.*}}, 0
  // CHECK %{{.*}} = extractvalue { <16 x float>, float } %{{.*}}, 1
  // CHECK %{{.*}} = call { <32 x float>, float } @llvm.nvvm.tcgen05.ld.red.16x32bx2.x32.f32(ptr addrspace(6) %{{.*}}, i64 0, i32 0, i1 true, i1 true)
  // CHECK %{{.*}} = extractvalue { <32 x float>, float } %{{.*}}, 0
  // CHECK %{{.*}} = extractvalue { <32 x float>, float } %{{.*}}, 1
  // CHECK %{{.*}} = call { <64 x float>, float } @llvm.nvvm.tcgen05.ld.red.16x32bx2.x64.f32(ptr addrspace(6) %{{.*}}, i64 0, i32 0, i1 true, i1 true)
  // CHECK %{{.*}} = extractvalue { <64 x float>, float } %{{.*}}, 0
  // CHECK %{{.*}} = extractvalue { <64 x float>, float } %{{.*}}, 1
  // CHECK %{{.*}} = call { <128 x float>, float } @llvm.nvvm.tcgen05.ld.red.16x32bx2.x128.f32(ptr addrspace(6) %{{.*}}, i64 0, i32 0, i1 true, i1 true)
  // CHECK %{{.*}} = extractvalue { <128 x float>, float } %{{.*}}, 0
  // CHECK %{{.*}} = extractvalue { <128 x float>, float } %{{.*}}, 1

  %data7, %redval7 = nvvm.tcgen05.ld.red min %addr, %offset { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>, nan, abs} : vector<2 x f32>, f32

  %data8, %redval8 = nvvm.tcgen05.ld.red min %addr, %offset { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>, nan, abs} : vector<4 x f32>, f32

  %data9, %redval9 = nvvm.tcgen05.ld.red min %addr, %offset { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>, nan, abs} : vector<8 x f32>, f32

  %data10, %redval10 = nvvm.tcgen05.ld.red min %addr, %offset { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>, nan, abs} : vector<16 x f32>, f32

  %data11, %redval11 = nvvm.tcgen05.ld.red min %addr, %offset { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>, nan, abs} : vector<32 x f32>, f32

  %data12, %redval12 = nvvm.tcgen05.ld.red min %addr, %offset { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>, nan, abs} : vector<64 x f32>, f32

  %data13, %redval13 = nvvm.tcgen05.ld.red min %addr, %offset { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>, nan, abs} : vector<128 x f32>, f32
  llvm.return
}

// CHECK-LABEL: @nvvm_tcgen05_ld_16x32bx2_max_nan_abs
llvm.func @nvvm_tcgen05_ld_16x32bx2_max_nan_abs(%addr : !llvm.ptr<6>) {

  %offset = llvm.mlir.constant(0: i64) : i64
  // CHECK %{{.*}} = call { <2 x float>, float } @llvm.nvvm.tcgen05.ld.red.16x32bx2.x2.f32(ptr addrspace(6) %{{.*}}, i64 0, i32 1, i1 true, i1 true)
  // CHECK %{{.*}} = extractvalue { <2 x float>, float } %{{.*}}, 0
  // CHECK %{{.*}} = extractvalue { <2 x float>, float } %{{.*}}, 1
  // CHECK %{{.*}} = call { <4 x float>, float } @llvm.nvvm.tcgen05.ld.red.16x32bx2.x4.f32(ptr addrspace(6) %{{.*}}, i64 0, i32 1, i1 true, i1 true)
  // CHECK %{{.*}} = extractvalue { <4 x float>, float } %{{.*}}, 0
  // CHECK %{{.*}} = extractvalue { <4 x float>, float } %{{.*}}, 1
  // CHECK %{{.*}} = call { <8 x float>, float } @llvm.nvvm.tcgen05.ld.red.16x32bx2.x8.f32(ptr addrspace(6) %{{.*}}, i64 0, i32 1, i1 true, i1 true)
  // CHECK %{{.*}} = extractvalue { <8 x float>, float } %{{.*}}, 0
  // CHECK %{{.*}} = extractvalue { <8 x float>, float } %{{.*}}, 1
  // CHECK %{{.*}} = call { <16 x float>, float } @llvm.nvvm.tcgen05.ld.red.16x32bx2.x16.f32(ptr addrspace(6) %{{.*}}, i64 0, i32 1, i1 true, i1 true)
  // CHECK %{{.*}} = extractvalue { <16 x float>, float } %{{.*}}, 0
  // CHECK %{{.*}} = extractvalue { <16 x float>, float } %{{.*}}, 1
  // CHECK %{{.*}} = call { <32 x float>, float } @llvm.nvvm.tcgen05.ld.red.16x32bx2.x32.f32(ptr addrspace(6) %{{.*}}, i64 0, i32 1, i1 true, i1 true)
  // CHECK %{{.*}} = extractvalue { <32 x float>, float } %{{.*}}, 0
  // CHECK %{{.*}} = extractvalue { <32 x float>, float } %{{.*}}, 1
  // CHECK %{{.*}} = call { <64 x float>, float } @llvm.nvvm.tcgen05.ld.red.16x32bx2.x64.f32(ptr addrspace(6) %{{.*}}, i64 0, i32 1, i1 true, i1 true)
  // CHECK %{{.*}} = extractvalue { <64 x float>, float } %{{.*}}, 0
  // CHECK %{{.*}} = extractvalue { <64 x float>, float } %{{.*}}, 1
  // CHECK %{{.*}} = call { <128 x float>, float } @llvm.nvvm.tcgen05.ld.red.16x32bx2.x128.f32(ptr addrspace(6) %{{.*}}, i64 0, i32 1, i1 true, i1 true)
  // CHECK %{{.*}} = extractvalue { <128 x float>, float } %{{.*}}, 0
  // CHECK %{{.*}} = extractvalue { <128 x float>, float } %{{.*}}, 1

  %data7, %redval7 = nvvm.tcgen05.ld.red max %addr, %offset { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>, nan, abs} : vector<2 x f32>, f32

  %data8, %redval8 = nvvm.tcgen05.ld.red max %addr, %offset { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>, nan, abs} : vector<4 x f32>, f32

  %data9, %redval9 = nvvm.tcgen05.ld.red max %addr, %offset { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>, nan, abs} : vector<8 x f32>, f32

  %data10, %redval10 = nvvm.tcgen05.ld.red max %addr, %offset { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>, nan, abs} : vector<16 x f32>, f32

  %data11, %redval11 = nvvm.tcgen05.ld.red max %addr, %offset { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>, nan, abs} : vector<32 x f32>, f32

  %data12, %redval12 = nvvm.tcgen05.ld.red max %addr, %offset { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>, nan, abs} : vector<64 x f32>, f32

  %data13, %redval13 = nvvm.tcgen05.ld.red max %addr, %offset { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>, nan, abs} : vector<128 x f32>, f32
  llvm.return
}
