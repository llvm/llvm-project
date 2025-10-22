// RUN: mlir-translate --mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: @nvvm_tcgen05_ld_16x64b
llvm.func @nvvm_tcgen05_ld_16x64b(%tmemAddr : !llvm.ptr<6>) {

// CHECK:  call i32 @llvm.nvvm.tcgen05.ld.16x64b.x1(ptr addrspace(6) {{%[0-9]+}}, i1 false)
  %ldv1 = nvvm.tcgen05.ld %tmemAddr { shape = #nvvm.tcgen05_ldst_shape<shape_16x64b>} : i32

// CHECK:  call <2 x i32> @llvm.nvvm.tcgen05.ld.16x64b.x2(ptr addrspace(6) {{%[0-9]+}}, i1 false)
  %ldv2 = nvvm.tcgen05.ld %tmemAddr { shape = #nvvm.tcgen05_ldst_shape<shape_16x64b>} : vector<2 x i32>

// CHECK:  call <4 x i32> @llvm.nvvm.tcgen05.ld.16x64b.x4(ptr addrspace(6) {{%[0-9]+}}, i1 false)
  %ldv4 = nvvm.tcgen05.ld %tmemAddr { shape = #nvvm.tcgen05_ldst_shape<shape_16x64b>} : vector<4 x i32>

// CHECK:  call <8 x i32> @llvm.nvvm.tcgen05.ld.16x64b.x8(ptr addrspace(6) {{%[0-9]+}}, i1 false)
  %ldv8 = nvvm.tcgen05.ld %tmemAddr { shape = #nvvm.tcgen05_ldst_shape<shape_16x64b>} : vector<8 x i32>

// CHECK:  call <16 x i32> @llvm.nvvm.tcgen05.ld.16x64b.x16(ptr addrspace(6) {{%[0-9]+}}, i1 false)
  %ldv16= nvvm.tcgen05.ld %tmemAddr { shape = #nvvm.tcgen05_ldst_shape<shape_16x64b>} : vector<16 x i32>

// CHECK:  call <32 x i32> @llvm.nvvm.tcgen05.ld.16x64b.x32(ptr addrspace(6) {{%[0-9]+}}, i1 false)
  %ldv32= nvvm.tcgen05.ld %tmemAddr { shape = #nvvm.tcgen05_ldst_shape<shape_16x64b>} : vector<32 x i32>

// CHECK:  call <64 x i32> @llvm.nvvm.tcgen05.ld.16x64b.x64(ptr addrspace(6) {{%[0-9]+}}, i1 false)
  %ldv64= nvvm.tcgen05.ld %tmemAddr { shape = #nvvm.tcgen05_ldst_shape<shape_16x64b>} : vector<64 x i32>

// CHECK:  call <128 x i32> @llvm.nvvm.tcgen05.ld.16x64b.x128(ptr addrspace(6) {{%[0-9]+}}, i1 false)
  %ldv128= nvvm.tcgen05.ld %tmemAddr { shape = #nvvm.tcgen05_ldst_shape<shape_16x64b>} : vector<128 x i32>

  llvm.return
}

// CHECK-LABEL: @nvvm_tcgen05_ld_16x64b_pack
llvm.func @nvvm_tcgen05_ld_16x64b_pack(%tmemAddr : !llvm.ptr<6>) {

// CHECK:  call i32 @llvm.nvvm.tcgen05.ld.16x64b.x1(ptr addrspace(6) {{%[0-9]+}}, i1 true)
  %ldv1 = nvvm.tcgen05.ld %tmemAddr pack { shape = #nvvm.tcgen05_ldst_shape<shape_16x64b>} : i32

// CHECK:  call <2 x i32> @llvm.nvvm.tcgen05.ld.16x64b.x2(ptr addrspace(6) {{%[0-9]+}}, i1 true)
  %ldv2 = nvvm.tcgen05.ld %tmemAddr pack { shape = #nvvm.tcgen05_ldst_shape<shape_16x64b>} : vector<2 x i32>

// CHECK:  call <4 x i32> @llvm.nvvm.tcgen05.ld.16x64b.x4(ptr addrspace(6) {{%[0-9]+}}, i1 true)
  %ldv4 = nvvm.tcgen05.ld %tmemAddr pack { shape = #nvvm.tcgen05_ldst_shape<shape_16x64b>} : vector<4 x i32>

// CHECK:  call <8 x i32> @llvm.nvvm.tcgen05.ld.16x64b.x8(ptr addrspace(6) {{%[0-9]+}}, i1 true)
  %ldv8 = nvvm.tcgen05.ld %tmemAddr pack { shape = #nvvm.tcgen05_ldst_shape<shape_16x64b>} : vector<8 x i32>

// CHECK:  call <16 x i32> @llvm.nvvm.tcgen05.ld.16x64b.x16(ptr addrspace(6) {{%[0-9]+}}, i1 true)
  %ldv16= nvvm.tcgen05.ld %tmemAddr pack { shape = #nvvm.tcgen05_ldst_shape<shape_16x64b>} : vector<16 x i32>

// CHECK:  call <32 x i32> @llvm.nvvm.tcgen05.ld.16x64b.x32(ptr addrspace(6) {{%[0-9]+}}, i1 true)
  %ldv32= nvvm.tcgen05.ld %tmemAddr pack { shape = #nvvm.tcgen05_ldst_shape<shape_16x64b>} : vector<32 x i32>

// CHECK:  call <64 x i32> @llvm.nvvm.tcgen05.ld.16x64b.x64(ptr addrspace(6) {{%[0-9]+}}, i1 true)
  %ldv64= nvvm.tcgen05.ld %tmemAddr pack { shape = #nvvm.tcgen05_ldst_shape<shape_16x64b>} : vector<64 x i32>

// CHECK:  call <128 x i32> @llvm.nvvm.tcgen05.ld.16x64b.x128(ptr addrspace(6) {{%[0-9]+}}, i1 true)
  %ldv128= nvvm.tcgen05.ld %tmemAddr pack { shape = #nvvm.tcgen05_ldst_shape<shape_16x64b>} : vector<128 x i32>

  llvm.return
}

// CHECK-LABEL: @nvvm_tcgen05_ld_16x128b
llvm.func @nvvm_tcgen05_ld_16x128b(%tmemAddr : !llvm.ptr<6>) {

// CHECK:  call <2 x i32> @llvm.nvvm.tcgen05.ld.16x128b.x1(ptr addrspace(6) {{%[0-9]+}}, i1 false)
  %ldv2 = nvvm.tcgen05.ld %tmemAddr { shape = #nvvm.tcgen05_ldst_shape<shape_16x128b>} : vector<2 x i32>

// CHECK:  call <4 x i32> @llvm.nvvm.tcgen05.ld.16x128b.x2(ptr addrspace(6) {{%[0-9]+}}, i1 false)
  %ldv4 = nvvm.tcgen05.ld %tmemAddr { shape = #nvvm.tcgen05_ldst_shape<shape_16x128b>} : vector<4 x i32>

// CHECK:  call <8 x i32> @llvm.nvvm.tcgen05.ld.16x128b.x4(ptr addrspace(6) {{%[0-9]+}}, i1 false)
  %ldv8 = nvvm.tcgen05.ld %tmemAddr { shape = #nvvm.tcgen05_ldst_shape<shape_16x128b>} : vector<8 x i32>

// CHECK:  call <16 x i32> @llvm.nvvm.tcgen05.ld.16x128b.x8(ptr addrspace(6) {{%[0-9]+}}, i1 false)
  %ldv16= nvvm.tcgen05.ld %tmemAddr { shape = #nvvm.tcgen05_ldst_shape<shape_16x128b>} : vector<16 x i32>

// CHECK:  call <32 x i32> @llvm.nvvm.tcgen05.ld.16x128b.x16(ptr addrspace(6) {{%[0-9]+}}, i1 false)
  %ldv32= nvvm.tcgen05.ld %tmemAddr { shape = #nvvm.tcgen05_ldst_shape<shape_16x128b>} : vector<32 x i32>

// CHECK:  call <64 x i32> @llvm.nvvm.tcgen05.ld.16x128b.x32(ptr addrspace(6) {{%[0-9]+}}, i1 false)
  %ldv64= nvvm.tcgen05.ld %tmemAddr { shape = #nvvm.tcgen05_ldst_shape<shape_16x128b>} : vector<64 x i32>

// CHECK:  call <128 x i32> @llvm.nvvm.tcgen05.ld.16x128b.x64(ptr addrspace(6) {{%[0-9]+}}, i1 false)
  %ldv128= nvvm.tcgen05.ld %tmemAddr { shape = #nvvm.tcgen05_ldst_shape<shape_16x128b>} : vector<128 x i32>

  llvm.return
}

// CHECK-LABEL: @nvvm_tcgen05_ld_16x128b_pack
llvm.func @nvvm_tcgen05_ld_16x128b_pack(%tmemAddr : !llvm.ptr<6>) {

// CHECK:  call <2 x i32> @llvm.nvvm.tcgen05.ld.16x128b.x1(ptr addrspace(6) {{%[0-9]+}}, i1 true)
  %ldv2 = nvvm.tcgen05.ld %tmemAddr pack { shape = #nvvm.tcgen05_ldst_shape<shape_16x128b>} : vector<2 x i32>

// CHECK:  call <4 x i32> @llvm.nvvm.tcgen05.ld.16x128b.x2(ptr addrspace(6) {{%[0-9]+}}, i1 true)
  %ldv4 = nvvm.tcgen05.ld %tmemAddr pack { shape = #nvvm.tcgen05_ldst_shape<shape_16x128b>} : vector<4 x i32>

// CHECK:  call <8 x i32> @llvm.nvvm.tcgen05.ld.16x128b.x4(ptr addrspace(6) {{%[0-9]+}}, i1 true)
  %ldv8 = nvvm.tcgen05.ld %tmemAddr pack { shape = #nvvm.tcgen05_ldst_shape<shape_16x128b>} : vector<8 x i32>

// CHECK:  call <16 x i32> @llvm.nvvm.tcgen05.ld.16x128b.x8(ptr addrspace(6) {{%[0-9]+}}, i1 true)
  %ldv16= nvvm.tcgen05.ld %tmemAddr pack { shape = #nvvm.tcgen05_ldst_shape<shape_16x128b>} : vector<16 x i32>

// CHECK:  call <32 x i32> @llvm.nvvm.tcgen05.ld.16x128b.x16(ptr addrspace(6) {{%[0-9]+}}, i1 true)
  %ldv32= nvvm.tcgen05.ld %tmemAddr pack { shape = #nvvm.tcgen05_ldst_shape<shape_16x128b>} : vector<32 x i32>

// CHECK:  call <64 x i32> @llvm.nvvm.tcgen05.ld.16x128b.x32(ptr addrspace(6) {{%[0-9]+}}, i1 true)
  %ldv64= nvvm.tcgen05.ld %tmemAddr pack { shape = #nvvm.tcgen05_ldst_shape<shape_16x128b>} : vector<64 x i32>

// CHECK:  call <128 x i32> @llvm.nvvm.tcgen05.ld.16x128b.x64(ptr addrspace(6) {{%[0-9]+}}, i1 true)
  %ldv128= nvvm.tcgen05.ld %tmemAddr pack { shape = #nvvm.tcgen05_ldst_shape<shape_16x128b>} : vector<128 x i32>

  llvm.return
}

// CHECK-LABEL: @nvvm_tcgen05_ld_16x256b
llvm.func @nvvm_tcgen05_ld_16x256b(%tmemAddr : !llvm.ptr<6>) {

// CHECK:  call <4 x i32> @llvm.nvvm.tcgen05.ld.16x256b.x1(ptr addrspace(6) {{%[0-9]+}}, i1 false)
  %ldv4 = nvvm.tcgen05.ld %tmemAddr { shape = #nvvm.tcgen05_ldst_shape<shape_16x256b>} : vector<4 x i32>

// CHECK:  call <8 x i32> @llvm.nvvm.tcgen05.ld.16x256b.x2(ptr addrspace(6) {{%[0-9]+}}, i1 false)
  %ldv8 = nvvm.tcgen05.ld %tmemAddr { shape = #nvvm.tcgen05_ldst_shape<shape_16x256b>} : vector<8 x i32>

// CHECK:  call <16 x i32> @llvm.nvvm.tcgen05.ld.16x256b.x4(ptr addrspace(6) {{%[0-9]+}}, i1 false)
  %ldv16= nvvm.tcgen05.ld %tmemAddr { shape = #nvvm.tcgen05_ldst_shape<shape_16x256b>} : vector<16 x i32>

// CHECK:  call <32 x i32> @llvm.nvvm.tcgen05.ld.16x256b.x8(ptr addrspace(6) {{%[0-9]+}}, i1 false)
  %ldv32= nvvm.tcgen05.ld %tmemAddr { shape = #nvvm.tcgen05_ldst_shape<shape_16x256b>} : vector<32 x i32>

// CHECK:  call <64 x i32> @llvm.nvvm.tcgen05.ld.16x256b.x16(ptr addrspace(6) {{%[0-9]+}}, i1 false)
  %ldv64= nvvm.tcgen05.ld %tmemAddr { shape = #nvvm.tcgen05_ldst_shape<shape_16x256b>} : vector<64 x i32>

// CHECK:  call <128 x i32> @llvm.nvvm.tcgen05.ld.16x256b.x32(ptr addrspace(6) {{%[0-9]+}}, i1 false)
  %ldv128= nvvm.tcgen05.ld %tmemAddr { shape = #nvvm.tcgen05_ldst_shape<shape_16x256b>} : vector<128 x i32>

  llvm.return
}

// CHECK-LABEL: @nvvm_tcgen05_ld_16x256b_pack
llvm.func @nvvm_tcgen05_ld_16x256b_pack(%tmemAddr : !llvm.ptr<6>) {

// CHECK:  call <4 x i32> @llvm.nvvm.tcgen05.ld.16x256b.x1(ptr addrspace(6) {{%[0-9]+}}, i1 true)
  %ldv4 = nvvm.tcgen05.ld %tmemAddr pack { shape = #nvvm.tcgen05_ldst_shape<shape_16x256b>} : vector<4 x i32>

// CHECK:  call <8 x i32> @llvm.nvvm.tcgen05.ld.16x256b.x2(ptr addrspace(6) {{%[0-9]+}}, i1 true)
  %ldv8 = nvvm.tcgen05.ld %tmemAddr pack { shape = #nvvm.tcgen05_ldst_shape<shape_16x256b>} : vector<8 x i32>

// CHECK:  call <16 x i32> @llvm.nvvm.tcgen05.ld.16x256b.x4(ptr addrspace(6) {{%[0-9]+}}, i1 true)
  %ldv16= nvvm.tcgen05.ld %tmemAddr pack { shape = #nvvm.tcgen05_ldst_shape<shape_16x256b>} : vector<16 x i32>

// CHECK:  call <32 x i32> @llvm.nvvm.tcgen05.ld.16x256b.x8(ptr addrspace(6) {{%[0-9]+}}, i1 true)
  %ldv32= nvvm.tcgen05.ld %tmemAddr pack { shape = #nvvm.tcgen05_ldst_shape<shape_16x256b>} : vector<32 x i32>

// CHECK:  call <64 x i32> @llvm.nvvm.tcgen05.ld.16x256b.x16(ptr addrspace(6) {{%[0-9]+}}, i1 true)
  %ldv64= nvvm.tcgen05.ld %tmemAddr pack { shape = #nvvm.tcgen05_ldst_shape<shape_16x256b>} : vector<64 x i32>

// CHECK:  call <128 x i32> @llvm.nvvm.tcgen05.ld.16x256b.x32(ptr addrspace(6) {{%[0-9]+}}, i1 true)
  %ldv128= nvvm.tcgen05.ld %tmemAddr pack { shape = #nvvm.tcgen05_ldst_shape<shape_16x256b>} : vector<128 x i32>

  llvm.return
}

// CHECK-LABEL: @nvvm_tcgen05_ld_32x32b
llvm.func @nvvm_tcgen05_ld_32x32b(%tmemAddr : !llvm.ptr<6>) {

// CHECK:  call i32 @llvm.nvvm.tcgen05.ld.32x32b.x1(ptr addrspace(6) {{%[0-9]+}}, i1 false)
  %ldv1 = nvvm.tcgen05.ld %tmemAddr { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>} : i32

// CHECK:  call <2 x i32> @llvm.nvvm.tcgen05.ld.32x32b.x2(ptr addrspace(6) {{%[0-9]+}}, i1 false)
  %ldv2 = nvvm.tcgen05.ld %tmemAddr { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>} : vector<2 x i32>

// CHECK:  call <4 x i32> @llvm.nvvm.tcgen05.ld.32x32b.x4(ptr addrspace(6) {{%[0-9]+}}, i1 false)
  %ldv4 = nvvm.tcgen05.ld %tmemAddr { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>} : vector<4 x i32>

// CHECK:  call <8 x i32> @llvm.nvvm.tcgen05.ld.32x32b.x8(ptr addrspace(6) {{%[0-9]+}}, i1 false)
  %ldv8 = nvvm.tcgen05.ld %tmemAddr { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>} : vector<8 x i32>

// CHECK:  call <16 x i32> @llvm.nvvm.tcgen05.ld.32x32b.x16(ptr addrspace(6) {{%[0-9]+}}, i1 false)
  %ldv16= nvvm.tcgen05.ld %tmemAddr { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>} : vector<16 x i32>

// CHECK:  call <32 x i32> @llvm.nvvm.tcgen05.ld.32x32b.x32(ptr addrspace(6) {{%[0-9]+}}, i1 false)
  %ldv32= nvvm.tcgen05.ld %tmemAddr { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>} : vector<32 x i32>

// CHECK:  call <64 x i32> @llvm.nvvm.tcgen05.ld.32x32b.x64(ptr addrspace(6) {{%[0-9]+}}, i1 false)
  %ldv64= nvvm.tcgen05.ld %tmemAddr { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>} : vector<64 x i32>

// CHECK:  call <128 x i32> @llvm.nvvm.tcgen05.ld.32x32b.x128(ptr addrspace(6) {{%[0-9]+}}, i1 false)
  %ldv128= nvvm.tcgen05.ld %tmemAddr { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>} : vector<128 x i32>

  llvm.return
}

// CHECK-LABEL: @nvvm_tcgen05_ld_32x32b_pack
llvm.func @nvvm_tcgen05_ld_32x32b_pack(%tmemAddr : !llvm.ptr<6>) {

// CHECK:  call i32 @llvm.nvvm.tcgen05.ld.32x32b.x1(ptr addrspace(6) {{%[0-9]+}}, i1 true)
  %ldv1 = nvvm.tcgen05.ld %tmemAddr pack { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>} : i32

// CHECK:  call <2 x i32> @llvm.nvvm.tcgen05.ld.32x32b.x2(ptr addrspace(6) {{%[0-9]+}}, i1 true)
  %ldv2 = nvvm.tcgen05.ld %tmemAddr pack { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>} : vector<2 x i32>

// CHECK:  call <4 x i32> @llvm.nvvm.tcgen05.ld.32x32b.x4(ptr addrspace(6) {{%[0-9]+}}, i1 true)
  %ldv4 = nvvm.tcgen05.ld %tmemAddr pack { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>} : vector<4 x i32>

// CHECK:  call <8 x i32> @llvm.nvvm.tcgen05.ld.32x32b.x8(ptr addrspace(6) {{%[0-9]+}}, i1 true)
  %ldv8 = nvvm.tcgen05.ld %tmemAddr pack { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>} : vector<8 x i32>

// CHECK:  call <16 x i32> @llvm.nvvm.tcgen05.ld.32x32b.x16(ptr addrspace(6) {{%[0-9]+}}, i1 true)
  %ldv16= nvvm.tcgen05.ld %tmemAddr pack { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>} : vector<16 x i32>

// CHECK:  call <32 x i32> @llvm.nvvm.tcgen05.ld.32x32b.x32(ptr addrspace(6) {{%[0-9]+}}, i1 true)
  %ldv32= nvvm.tcgen05.ld %tmemAddr pack { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>} : vector<32 x i32>

// CHECK:  call <64 x i32> @llvm.nvvm.tcgen05.ld.32x32b.x64(ptr addrspace(6) {{%[0-9]+}}, i1 true)
  %ldv64= nvvm.tcgen05.ld %tmemAddr pack { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>} : vector<64 x i32>

// CHECK:  call <128 x i32> @llvm.nvvm.tcgen05.ld.32x32b.x128(ptr addrspace(6) {{%[0-9]+}}, i1 true)
  %ldv128= nvvm.tcgen05.ld %tmemAddr pack { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>} : vector<128 x i32>

  llvm.return
}

// CHECK-LABEL: @nvvm_tcgen05_ld_16x32bx2
llvm.func @nvvm_tcgen05_ld_16x32bx2(%tmemAddr : !llvm.ptr<6>) {

  %halfSplitOffset = llvm.mlir.constant(2:i64) : i64

// CHECK:  call i32 @llvm.nvvm.tcgen05.ld.16x32bx2.x1(ptr addrspace(6) {{%[0-9]+}}, i64 2, i1 false)
  %ldv1 = nvvm.tcgen05.ld %tmemAddr, %halfSplitOffset { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>} : i32

// CHECK:  call <2 x i32> @llvm.nvvm.tcgen05.ld.16x32bx2.x2(ptr addrspace(6) {{%[0-9]+}}, i64 2, i1 false)
  %ldv2 = nvvm.tcgen05.ld %tmemAddr, %halfSplitOffset { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>} : vector<2 x i32>

// CHECK:  call <4 x i32> @llvm.nvvm.tcgen05.ld.16x32bx2.x4(ptr addrspace(6) {{%[0-9]+}}, i64 2, i1 false)
  %ldv4 = nvvm.tcgen05.ld %tmemAddr, %halfSplitOffset { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>} : vector<4 x i32>

// CHECK:  call <8 x i32> @llvm.nvvm.tcgen05.ld.16x32bx2.x8(ptr addrspace(6) {{%[0-9]+}}, i64 2, i1 false)
  %ldv8 = nvvm.tcgen05.ld %tmemAddr, %halfSplitOffset { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>} : vector<8 x i32>

// CHECK:  call <16 x i32> @llvm.nvvm.tcgen05.ld.16x32bx2.x16(ptr addrspace(6) {{%[0-9]+}}, i64 2, i1 false)
  %ldv16= nvvm.tcgen05.ld %tmemAddr, %halfSplitOffset { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>} : vector<16 x i32>

// CHECK:  call <32 x i32> @llvm.nvvm.tcgen05.ld.16x32bx2.x32(ptr addrspace(6) {{%[0-9]+}}, i64 2, i1 false)
  %ldv32= nvvm.tcgen05.ld %tmemAddr, %halfSplitOffset { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>} : vector<32 x i32>

// CHECK:  call <64 x i32> @llvm.nvvm.tcgen05.ld.16x32bx2.x64(ptr addrspace(6) {{%[0-9]+}}, i64 2, i1 false)
  %ldv64= nvvm.tcgen05.ld %tmemAddr, %halfSplitOffset { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>} : vector<64 x i32>

// CHECK:  call <128 x i32> @llvm.nvvm.tcgen05.ld.16x32bx2.x128(ptr addrspace(6) {{%[0-9]+}}, i64 2, i1 false)
  %ldv128= nvvm.tcgen05.ld %tmemAddr, %halfSplitOffset { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>} : vector<128 x i32>

  llvm.return
}

// CHECK-LABEL: @nvvm_tcgen05_ld_16x32bx2_pack
llvm.func @nvvm_tcgen05_ld_16x32bx2_pack(%tmemAddr : !llvm.ptr<6>) {

  %halfSplitOffset = llvm.mlir.constant(2:i64) : i64

// CHECK:  call i32 @llvm.nvvm.tcgen05.ld.16x32bx2.x1(ptr addrspace(6) {{%[0-9]+}}, i64 2, i1 true)
  %ldv1 = nvvm.tcgen05.ld %tmemAddr, %halfSplitOffset pack { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>} : i32

// CHECK:  call <2 x i32> @llvm.nvvm.tcgen05.ld.16x32bx2.x2(ptr addrspace(6) {{%[0-9]+}}, i64 2, i1 true)
  %ldv2 = nvvm.tcgen05.ld %tmemAddr, %halfSplitOffset pack { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>} : vector<2 x i32>

// CHECK:  call <4 x i32> @llvm.nvvm.tcgen05.ld.16x32bx2.x4(ptr addrspace(6) {{%[0-9]+}}, i64 2, i1 true)
  %ldv4 = nvvm.tcgen05.ld %tmemAddr, %halfSplitOffset pack { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>} : vector<4 x i32>

// CHECK:  call <8 x i32> @llvm.nvvm.tcgen05.ld.16x32bx2.x8(ptr addrspace(6) {{%[0-9]+}}, i64 2, i1 true)
  %ldv8 = nvvm.tcgen05.ld %tmemAddr, %halfSplitOffset pack { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>} : vector<8 x i32>

// CHECK:  call <16 x i32> @llvm.nvvm.tcgen05.ld.16x32bx2.x16(ptr addrspace(6) {{%[0-9]+}}, i64 2, i1 true)
  %ldv16= nvvm.tcgen05.ld %tmemAddr, %halfSplitOffset pack { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>} : vector<16 x i32>

// CHECK:  call <32 x i32> @llvm.nvvm.tcgen05.ld.16x32bx2.x32(ptr addrspace(6) {{%[0-9]+}}, i64 2, i1 true)
  %ldv32= nvvm.tcgen05.ld %tmemAddr, %halfSplitOffset pack { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>} : vector<32 x i32>

// CHECK:  call <64 x i32> @llvm.nvvm.tcgen05.ld.16x32bx2.x64(ptr addrspace(6) {{%[0-9]+}}, i64 2, i1 true)
  %ldv64= nvvm.tcgen05.ld %tmemAddr, %halfSplitOffset pack { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>} : vector<64 x i32>

// CHECK:  call <128 x i32> @llvm.nvvm.tcgen05.ld.16x32bx2.x128(ptr addrspace(6) {{%[0-9]+}}, i64 2, i1 true)
  %ldv128= nvvm.tcgen05.ld %tmemAddr, %halfSplitOffset pack { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>} : vector<128 x i32>

  llvm.return
}
