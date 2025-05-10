// RUN: mlir-translate --mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: @nvvm_tcgen05_ld_16x64b
llvm.func @nvvm_tcgen05_ld_16x64b(
  %tmemAddr : !llvm.ptr<6>,
  %stv1     : i32,
  %stv2     : vector<2xi32>,
  %stv4     : vector<4xi32>,
  %stv8     : vector<8xi32>,
  %stv16    : vector<16xi32>,
  %stv32    : vector<32xi32>,
  %stv64    : vector<64xi32>,
  %stv128   : vector<128xi32>) {

// CHECK: call void @llvm.nvvm.tcgen05.st.16x64b.x1(ptr addrspace(6) {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 false)
  nvvm.tcgen05.st %tmemAddr, %stv1 { shape = #nvvm.tcgen05_ldst_shape<shape_16x64b>, num=1:i32 } : i32

// CHECK: call void @llvm.nvvm.tcgen05.st.16x64b.x2(ptr addrspace(6) {{%[0-9]+}}, <2 x i32> {{%[0-9]+}}, i1 false)
  nvvm.tcgen05.st %tmemAddr, %stv2 { shape = #nvvm.tcgen05_ldst_shape<shape_16x64b>, num=2:i32 } : vector<2xi32>

// CHECK: call void @llvm.nvvm.tcgen05.st.16x64b.x4(ptr addrspace(6) {{%[0-9]+}}, <4 x i32> {{%[0-9]+}}, i1 false)
  nvvm.tcgen05.st %tmemAddr, %stv4 { shape = #nvvm.tcgen05_ldst_shape<shape_16x64b>, num=4:i32 } : vector<4xi32>

// CHECK: call void @llvm.nvvm.tcgen05.st.16x64b.x8(ptr addrspace(6) {{%[0-9]+}}, <8 x i32> {{%[0-9]+}}, i1 false)
  nvvm.tcgen05.st %tmemAddr, %stv8 { shape = #nvvm.tcgen05_ldst_shape<shape_16x64b>, num=8:i32 } : vector<8xi32>

// CHECK: call void @llvm.nvvm.tcgen05.st.16x64b.x16(ptr addrspace(6) {{%[0-9]+}}, <16 x i32> {{%[0-9]+}}, i1 false)
  nvvm.tcgen05.st %tmemAddr, %stv16 { shape = #nvvm.tcgen05_ldst_shape<shape_16x64b>, num=16:i32 } : vector<16xi32>

// CHECK: call void @llvm.nvvm.tcgen05.st.16x64b.x32(ptr addrspace(6) {{%[0-9]+}}, <32 x i32> {{%[0-9]+}}, i1 false)
  nvvm.tcgen05.st %tmemAddr, %stv32 { shape = #nvvm.tcgen05_ldst_shape<shape_16x64b>, num=32:i32 } : vector<32xi32>

// CHECK: call void @llvm.nvvm.tcgen05.st.16x64b.x64(ptr addrspace(6) {{%[0-9]+}}, <64 x i32> {{%[0-9]+}}, i1 false)
  nvvm.tcgen05.st %tmemAddr, %stv64 { shape = #nvvm.tcgen05_ldst_shape<shape_16x64b>, num=64:i32 } : vector<64xi32>

// CHECK: call void @llvm.nvvm.tcgen05.st.16x64b.x128(ptr addrspace(6) {{%[0-9]+}}, <128 x i32> {{%[0-9]+}}, i1 false)
  nvvm.tcgen05.st %tmemAddr, %stv128 { shape = #nvvm.tcgen05_ldst_shape<shape_16x64b>, num=128:i32 } : vector<128xi32>

  llvm.return
}

// CHECK-LABEL: @nvvm_tcgen05_ld_16x64b_pack
llvm.func @nvvm_tcgen05_ld_16x64b_pack(
  %tmemAddr : !llvm.ptr<6>,
  %stv1     : i32,
  %stv2     : vector<2xi32>,
  %stv4     : vector<4xi32>,
  %stv8     : vector<8xi32>,
  %stv16    : vector<16xi32>,
  %stv32    : vector<32xi32>,
  %stv64    : vector<64xi32>,
  %stv128   : vector<128xi32>) {

// CHECK: call void @llvm.nvvm.tcgen05.st.16x64b.x1(ptr addrspace(6) {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 true)
  nvvm.tcgen05.st %tmemAddr, %stv1 unpack { shape = #nvvm.tcgen05_ldst_shape<shape_16x64b>, num=1:i32 } : i32

// CHECK: call void @llvm.nvvm.tcgen05.st.16x64b.x2(ptr addrspace(6) {{%[0-9]+}}, <2 x i32> {{%[0-9]+}}, i1 true)
  nvvm.tcgen05.st %tmemAddr, %stv2 unpack { shape = #nvvm.tcgen05_ldst_shape<shape_16x64b>, num=2:i32 } : vector<2xi32>

// CHECK: call void @llvm.nvvm.tcgen05.st.16x64b.x4(ptr addrspace(6) {{%[0-9]+}}, <4 x i32> {{%[0-9]+}}, i1 true)
  nvvm.tcgen05.st %tmemAddr, %stv4 unpack { shape = #nvvm.tcgen05_ldst_shape<shape_16x64b>, num=4:i32 } : vector<4xi32>

// CHECK: call void @llvm.nvvm.tcgen05.st.16x64b.x8(ptr addrspace(6) {{%[0-9]+}}, <8 x i32> {{%[0-9]+}}, i1 true)
  nvvm.tcgen05.st %tmemAddr, %stv8 unpack { shape = #nvvm.tcgen05_ldst_shape<shape_16x64b>, num=8:i32 } : vector<8xi32>

// CHECK: call void @llvm.nvvm.tcgen05.st.16x64b.x16(ptr addrspace(6) {{%[0-9]+}}, <16 x i32> {{%[0-9]+}}, i1 true)
  nvvm.tcgen05.st %tmemAddr, %stv16 unpack { shape = #nvvm.tcgen05_ldst_shape<shape_16x64b>, num=16:i32 } : vector<16xi32>

// CHECK: call void @llvm.nvvm.tcgen05.st.16x64b.x32(ptr addrspace(6) {{%[0-9]+}}, <32 x i32> {{%[0-9]+}}, i1 true)
  nvvm.tcgen05.st %tmemAddr, %stv32 unpack { shape = #nvvm.tcgen05_ldst_shape<shape_16x64b>, num=32:i32 } : vector<32xi32>

// CHECK: call void @llvm.nvvm.tcgen05.st.16x64b.x64(ptr addrspace(6) {{%[0-9]+}}, <64 x i32> {{%[0-9]+}}, i1 true)
  nvvm.tcgen05.st %tmemAddr, %stv64 unpack { shape = #nvvm.tcgen05_ldst_shape<shape_16x64b>, num=64:i32 } : vector<64xi32>

// CHECK: call void @llvm.nvvm.tcgen05.st.16x64b.x128(ptr addrspace(6) {{%[0-9]+}}, <128 x i32> {{%[0-9]+}}, i1 true)
  nvvm.tcgen05.st %tmemAddr, %stv128 unpack { shape = #nvvm.tcgen05_ldst_shape<shape_16x64b>, num=128:i32 } : vector<128xi32>

  llvm.return
}

// CHECK-LABEL: @nvvm_tcgen05_ld_16x128b
llvm.func @nvvm_tcgen05_ld_16x128b(
  %tmemAddr : !llvm.ptr<6>,
  %stv1     : i32,
  %stv2     : vector<2xi32>,
  %stv4     : vector<4xi32>,
  %stv8     : vector<8xi32>,
  %stv16    : vector<16xi32>,
  %stv32    : vector<32xi32>,
  %stv64    : vector<64xi32>,
  %stv128   : vector<128xi32>) {

// CHECK: call void @llvm.nvvm.tcgen05.st.16x128b.x1(ptr addrspace(6) {{%[0-9]+}}, <2 x i32> {{%[0-9]+}}, i1 false)
  nvvm.tcgen05.st %tmemAddr, %stv2 { shape = #nvvm.tcgen05_ldst_shape<shape_16x128b>, num=1:i32 } : vector<2xi32>

// CHECK: call void @llvm.nvvm.tcgen05.st.16x128b.x2(ptr addrspace(6) {{%[0-9]+}}, <4 x i32> {{%[0-9]+}}, i1 false)
  nvvm.tcgen05.st %tmemAddr, %stv4 { shape = #nvvm.tcgen05_ldst_shape<shape_16x128b>, num=2:i32 } : vector<4xi32>

// CHECK: call void @llvm.nvvm.tcgen05.st.16x128b.x4(ptr addrspace(6) {{%[0-9]+}}, <8 x i32> {{%[0-9]+}}, i1 false)
  nvvm.tcgen05.st %tmemAddr, %stv8 { shape = #nvvm.tcgen05_ldst_shape<shape_16x128b>, num=4:i32 } : vector<8xi32>

// CHECK: call void @llvm.nvvm.tcgen05.st.16x128b.x8(ptr addrspace(6) {{%[0-9]+}}, <16 x i32> {{%[0-9]+}}, i1 false)
  nvvm.tcgen05.st %tmemAddr, %stv16 { shape = #nvvm.tcgen05_ldst_shape<shape_16x128b>, num=8:i32 } : vector<16xi32>

// CHECK: call void @llvm.nvvm.tcgen05.st.16x128b.x16(ptr addrspace(6) {{%[0-9]+}}, <32 x i32> {{%[0-9]+}}, i1 false)
  nvvm.tcgen05.st %tmemAddr, %stv32 { shape = #nvvm.tcgen05_ldst_shape<shape_16x128b>, num=16:i32 } : vector<32xi32>

// CHECK: call void @llvm.nvvm.tcgen05.st.16x128b.x32(ptr addrspace(6) {{%[0-9]+}}, <64 x i32> {{%[0-9]+}}, i1 false)
  nvvm.tcgen05.st %tmemAddr, %stv64 { shape = #nvvm.tcgen05_ldst_shape<shape_16x128b>, num=32:i32 } : vector<64xi32>

// CHECK: call void @llvm.nvvm.tcgen05.st.16x128b.x64(ptr addrspace(6) {{%[0-9]+}}, <128 x i32> {{%[0-9]+}}, i1 false)
  nvvm.tcgen05.st %tmemAddr, %stv128 { shape = #nvvm.tcgen05_ldst_shape<shape_16x128b>, num=64:i32 } : vector<128xi32>

  llvm.return
}

// CHECK-LABEL: @nvvm_tcgen05_ld_16x128b_pack
llvm.func @nvvm_tcgen05_ld_16x128b_pack(
  %tmemAddr : !llvm.ptr<6>,
  %stv1     : i32,
  %stv2     : vector<2xi32>,
  %stv4     : vector<4xi32>,
  %stv8     : vector<8xi32>,
  %stv16    : vector<16xi32>,
  %stv32    : vector<32xi32>,
  %stv64    : vector<64xi32>,
  %stv128   : vector<128xi32>) {

// CHECK: call void @llvm.nvvm.tcgen05.st.16x128b.x1(ptr addrspace(6) {{%[0-9]+}}, <2 x i32> {{%[0-9]+}}, i1 true)
  nvvm.tcgen05.st %tmemAddr, %stv2 unpack { shape = #nvvm.tcgen05_ldst_shape<shape_16x128b>, num=1:i32 } : vector<2xi32>

// CHECK: call void @llvm.nvvm.tcgen05.st.16x128b.x2(ptr addrspace(6) {{%[0-9]+}}, <4 x i32> {{%[0-9]+}}, i1 true)
  nvvm.tcgen05.st %tmemAddr, %stv4 unpack { shape = #nvvm.tcgen05_ldst_shape<shape_16x128b>, num=2:i32 } : vector<4xi32>

// CHECK: call void @llvm.nvvm.tcgen05.st.16x128b.x4(ptr addrspace(6) {{%[0-9]+}}, <8 x i32> {{%[0-9]+}}, i1 true)
  nvvm.tcgen05.st %tmemAddr, %stv8 unpack { shape = #nvvm.tcgen05_ldst_shape<shape_16x128b>, num=4:i32 } : vector<8xi32>

// CHECK: call void @llvm.nvvm.tcgen05.st.16x128b.x8(ptr addrspace(6) {{%[0-9]+}}, <16 x i32> {{%[0-9]+}}, i1 true)
  nvvm.tcgen05.st %tmemAddr, %stv16 unpack { shape = #nvvm.tcgen05_ldst_shape<shape_16x128b>, num=8:i32 } : vector<16xi32>

// CHECK: call void @llvm.nvvm.tcgen05.st.16x128b.x16(ptr addrspace(6) {{%[0-9]+}}, <32 x i32> {{%[0-9]+}}, i1 true)
  nvvm.tcgen05.st %tmemAddr, %stv32 unpack { shape = #nvvm.tcgen05_ldst_shape<shape_16x128b>, num=16:i32 } : vector<32xi32>

// CHECK: call void @llvm.nvvm.tcgen05.st.16x128b.x32(ptr addrspace(6) {{%[0-9]+}}, <64 x i32> {{%[0-9]+}}, i1 true)
  nvvm.tcgen05.st %tmemAddr, %stv64 unpack { shape = #nvvm.tcgen05_ldst_shape<shape_16x128b>, num=32:i32 } : vector<64xi32>

// CHECK: call void @llvm.nvvm.tcgen05.st.16x128b.x64(ptr addrspace(6) {{%[0-9]+}}, <128 x i32> {{%[0-9]+}}, i1 true)
  nvvm.tcgen05.st %tmemAddr, %stv128 unpack { shape = #nvvm.tcgen05_ldst_shape<shape_16x128b>, num=64:i32 } : vector<128xi32>

  llvm.return
}

// CHECK-LABEL: @nvvm_tcgen05_ld_16x256b
llvm.func @nvvm_tcgen05_ld_16x256b(
  %tmemAddr : !llvm.ptr<6>,
  %stv1     : i32,
  %stv2     : vector<2xi32>,
  %stv4     : vector<4xi32>,
  %stv8     : vector<8xi32>,
  %stv16    : vector<16xi32>,
  %stv32    : vector<32xi32>,
  %stv64    : vector<64xi32>,
  %stv128   : vector<128xi32>) {

// CHECK: call void @llvm.nvvm.tcgen05.st.16x256b.x1(ptr addrspace(6) {{%[0-9]+}}, <4 x i32> {{%[0-9]+}}, i1 false)
  nvvm.tcgen05.st %tmemAddr, %stv4 { shape = #nvvm.tcgen05_ldst_shape<shape_16x256b>, num=1:i32 } : vector<4xi32>

// CHECK: call void @llvm.nvvm.tcgen05.st.16x256b.x2(ptr addrspace(6) {{%[0-9]+}}, <8 x i32> {{%[0-9]+}}, i1 false)
  nvvm.tcgen05.st %tmemAddr, %stv8 { shape = #nvvm.tcgen05_ldst_shape<shape_16x256b>, num=2:i32 } : vector<8xi32>

// CHECK: call void @llvm.nvvm.tcgen05.st.16x256b.x4(ptr addrspace(6) {{%[0-9]+}}, <16 x i32> {{%[0-9]+}}, i1 false)
  nvvm.tcgen05.st %tmemAddr, %stv16 { shape = #nvvm.tcgen05_ldst_shape<shape_16x256b>, num=4:i32 } : vector<16xi32>

// CHECK: call void @llvm.nvvm.tcgen05.st.16x256b.x8(ptr addrspace(6) {{%[0-9]+}}, <32 x i32> {{%[0-9]+}}, i1 false)
  nvvm.tcgen05.st %tmemAddr, %stv32 { shape = #nvvm.tcgen05_ldst_shape<shape_16x256b>, num=8:i32 } : vector<32xi32>

// CHECK: call void @llvm.nvvm.tcgen05.st.16x256b.x16(ptr addrspace(6) {{%[0-9]+}}, <64 x i32> {{%[0-9]+}}, i1 false)
  nvvm.tcgen05.st %tmemAddr, %stv64 { shape = #nvvm.tcgen05_ldst_shape<shape_16x256b>, num=16:i32 } : vector<64xi32>

// CHECK: call void @llvm.nvvm.tcgen05.st.16x256b.x32(ptr addrspace(6) {{%[0-9]+}}, <128 x i32> {{%[0-9]+}}, i1 false)
  nvvm.tcgen05.st %tmemAddr, %stv128 { shape = #nvvm.tcgen05_ldst_shape<shape_16x256b>, num=32:i32 } : vector<128xi32>

  llvm.return
}

// CHECK-LABEL: @nvvm_tcgen05_ld_16x256b_pack
llvm.func @nvvm_tcgen05_ld_16x256b_pack(
  %tmemAddr : !llvm.ptr<6>,
  %stv1     : i32,
  %stv2     : vector<2xi32>,
  %stv4     : vector<4xi32>,
  %stv8     : vector<8xi32>,
  %stv16    : vector<16xi32>,
  %stv32    : vector<32xi32>,
  %stv64    : vector<64xi32>,
  %stv128   : vector<128xi32>) {

// CHECK: call void @llvm.nvvm.tcgen05.st.16x256b.x1(ptr addrspace(6) {{%[0-9]+}}, <4 x i32> {{%[0-9]+}}, i1 true)
  nvvm.tcgen05.st %tmemAddr, %stv4 unpack { shape = #nvvm.tcgen05_ldst_shape<shape_16x256b>, num=1:i32 } : vector<4xi32>

// CHECK: call void @llvm.nvvm.tcgen05.st.16x256b.x2(ptr addrspace(6) {{%[0-9]+}}, <8 x i32> {{%[0-9]+}}, i1 true)
  nvvm.tcgen05.st %tmemAddr, %stv8 unpack { shape = #nvvm.tcgen05_ldst_shape<shape_16x256b>, num=2:i32 } : vector<8xi32>

// CHECK: call void @llvm.nvvm.tcgen05.st.16x256b.x4(ptr addrspace(6) {{%[0-9]+}}, <16 x i32> {{%[0-9]+}}, i1 true)
  nvvm.tcgen05.st %tmemAddr, %stv16 unpack { shape = #nvvm.tcgen05_ldst_shape<shape_16x256b>, num=4:i32 } : vector<16xi32>

// CHECK: call void @llvm.nvvm.tcgen05.st.16x256b.x8(ptr addrspace(6) {{%[0-9]+}}, <32 x i32> {{%[0-9]+}}, i1 true)
  nvvm.tcgen05.st %tmemAddr, %stv32 unpack { shape = #nvvm.tcgen05_ldst_shape<shape_16x256b>, num=8:i32 } : vector<32xi32>

// CHECK: call void @llvm.nvvm.tcgen05.st.16x256b.x16(ptr addrspace(6) {{%[0-9]+}}, <64 x i32> {{%[0-9]+}}, i1 true)
  nvvm.tcgen05.st %tmemAddr, %stv64 unpack { shape = #nvvm.tcgen05_ldst_shape<shape_16x256b>, num=16:i32 } : vector<64xi32>

// CHECK: call void @llvm.nvvm.tcgen05.st.16x256b.x32(ptr addrspace(6) {{%[0-9]+}}, <128 x i32> {{%[0-9]+}}, i1 true)
  nvvm.tcgen05.st %tmemAddr, %stv128 unpack { shape = #nvvm.tcgen05_ldst_shape<shape_16x256b>, num=32:i32 } : vector<128xi32>

  llvm.return
}

// CHECK-LABEL: @nvvm_tcgen05_ld_32x32b
llvm.func @nvvm_tcgen05_ld_32x32b(
  %tmemAddr : !llvm.ptr<6>,
  %stv1     : i32,
  %stv2     : vector<2xi32>,
  %stv4     : vector<4xi32>,
  %stv8     : vector<8xi32>,
  %stv16    : vector<16xi32>,
  %stv32    : vector<32xi32>,
  %stv64    : vector<64xi32>,
  %stv128   : vector<128xi32>) {

// CHECK: call void @llvm.nvvm.tcgen05.st.32x32b.x1(ptr addrspace(6) {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 false)
  nvvm.tcgen05.st %tmemAddr, %stv1 { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>, num=1:i32 } : i32

// CHECK: call void @llvm.nvvm.tcgen05.st.32x32b.x2(ptr addrspace(6) {{%[0-9]+}}, <2 x i32> {{%[0-9]+}}, i1 false)
  nvvm.tcgen05.st %tmemAddr, %stv2 { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>, num=2:i32 } : vector<2xi32>

// CHECK: call void @llvm.nvvm.tcgen05.st.32x32b.x4(ptr addrspace(6) {{%[0-9]+}}, <4 x i32> {{%[0-9]+}}, i1 false)
  nvvm.tcgen05.st %tmemAddr, %stv4 { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>, num=4:i32 } : vector<4xi32>

// CHECK: call void @llvm.nvvm.tcgen05.st.32x32b.x8(ptr addrspace(6) {{%[0-9]+}}, <8 x i32> {{%[0-9]+}}, i1 false)
  nvvm.tcgen05.st %tmemAddr, %stv8 { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>, num=8:i32 } : vector<8xi32>

// CHECK: call void @llvm.nvvm.tcgen05.st.32x32b.x16(ptr addrspace(6) {{%[0-9]+}}, <16 x i32> {{%[0-9]+}}, i1 false)
  nvvm.tcgen05.st %tmemAddr, %stv16 { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>, num=16:i32 } : vector<16xi32>

// CHECK: call void @llvm.nvvm.tcgen05.st.32x32b.x32(ptr addrspace(6) {{%[0-9]+}}, <32 x i32> {{%[0-9]+}}, i1 false)
  nvvm.tcgen05.st %tmemAddr, %stv32 { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>, num=32:i32 } : vector<32xi32>

// CHECK: call void @llvm.nvvm.tcgen05.st.32x32b.x64(ptr addrspace(6) {{%[0-9]+}}, <64 x i32> {{%[0-9]+}}, i1 false)
  nvvm.tcgen05.st %tmemAddr, %stv64 { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>, num=64:i32 } : vector<64xi32>

// CHECK: call void @llvm.nvvm.tcgen05.st.32x32b.x128(ptr addrspace(6) {{%[0-9]+}}, <128 x i32> {{%[0-9]+}}, i1 false)
  nvvm.tcgen05.st %tmemAddr, %stv128 { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>, num=128:i32 } : vector<128xi32>

  llvm.return
}

// CHECK-LABEL: @nvvm_tcgen05_ld_32x32b_pack
llvm.func @nvvm_tcgen05_ld_32x32b_pack(
  %tmemAddr : !llvm.ptr<6>,
  %stv1     : i32,
  %stv2     : vector<2xi32>,
  %stv4     : vector<4xi32>,
  %stv8     : vector<8xi32>,
  %stv16    : vector<16xi32>,
  %stv32    : vector<32xi32>,
  %stv64    : vector<64xi32>,
  %stv128   : vector<128xi32>) {

// CHECK: call void @llvm.nvvm.tcgen05.st.32x32b.x1(ptr addrspace(6) {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 true)
  nvvm.tcgen05.st %tmemAddr, %stv1 unpack { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>, num=1:i32 } : i32

// CHECK: call void @llvm.nvvm.tcgen05.st.32x32b.x2(ptr addrspace(6) {{%[0-9]+}}, <2 x i32> {{%[0-9]+}}, i1 true)
  nvvm.tcgen05.st %tmemAddr, %stv2 unpack { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>, num=2:i32 } : vector<2xi32>

// CHECK: call void @llvm.nvvm.tcgen05.st.32x32b.x4(ptr addrspace(6) {{%[0-9]+}}, <4 x i32> {{%[0-9]+}}, i1 true)
  nvvm.tcgen05.st %tmemAddr, %stv4 unpack { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>, num=4:i32 } : vector<4xi32>

// CHECK: call void @llvm.nvvm.tcgen05.st.32x32b.x8(ptr addrspace(6) {{%[0-9]+}}, <8 x i32> {{%[0-9]+}}, i1 true)
  nvvm.tcgen05.st %tmemAddr, %stv8 unpack { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>, num=8:i32 } : vector<8xi32>

// CHECK: call void @llvm.nvvm.tcgen05.st.32x32b.x16(ptr addrspace(6) {{%[0-9]+}}, <16 x i32> {{%[0-9]+}}, i1 true)
  nvvm.tcgen05.st %tmemAddr, %stv16 unpack { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>, num=16:i32 } : vector<16xi32>

// CHECK: call void @llvm.nvvm.tcgen05.st.32x32b.x32(ptr addrspace(6) {{%[0-9]+}}, <32 x i32> {{%[0-9]+}}, i1 true)
  nvvm.tcgen05.st %tmemAddr, %stv32 unpack { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>, num=32:i32 } : vector<32xi32>

// CHECK: call void @llvm.nvvm.tcgen05.st.32x32b.x64(ptr addrspace(6) {{%[0-9]+}}, <64 x i32> {{%[0-9]+}}, i1 true)
  nvvm.tcgen05.st %tmemAddr, %stv64 unpack { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>, num=64:i32 } : vector<64xi32>

// CHECK: call void @llvm.nvvm.tcgen05.st.32x32b.x128(ptr addrspace(6) {{%[0-9]+}}, <128 x i32> {{%[0-9]+}}, i1 true)
  nvvm.tcgen05.st %tmemAddr, %stv128 unpack { shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>, num=128:i32 } : vector<128xi32>

  llvm.return
}

// CHECK-LABEL: @nvvm_tcgen05_ld_16x32bx2
llvm.func @nvvm_tcgen05_ld_16x32bx2(
  %tmemAddr : !llvm.ptr<6>,
  %stv1     : i32,
  %stv2     : vector<2xi32>,
  %stv4     : vector<4xi32>,
  %stv8     : vector<8xi32>,
  %stv16    : vector<16xi32>,
  %stv32    : vector<32xi32>,
  %stv64    : vector<64xi32>,
  %stv128   : vector<128xi32>) {

  %offset = llvm.mlir.constant(2:i64) : i64

// CHECK: call void @llvm.nvvm.tcgen05.st.16x32bx2.x1(ptr addrspace(6) {{%[0-9]+}}, i64 2, i32 {{%[0-9]+}}, i1 false)
  nvvm.tcgen05.st %tmemAddr, %stv1, %offset { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>, num=1:i32 } : i32

// CHECK: call void @llvm.nvvm.tcgen05.st.16x32bx2.x2(ptr addrspace(6) {{%[0-9]+}}, i64 2, <2 x i32> {{%[0-9]+}}, i1 false)
  nvvm.tcgen05.st %tmemAddr, %stv2, %offset { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>, num=2:i32 } : vector<2xi32>

// CHECK: call void @llvm.nvvm.tcgen05.st.16x32bx2.x4(ptr addrspace(6) {{%[0-9]+}}, i64 2, <4 x i32> {{%[0-9]+}}, i1 false)
  nvvm.tcgen05.st %tmemAddr, %stv4, %offset { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>, num=4:i32 } : vector<4xi32>

// CHECK: call void @llvm.nvvm.tcgen05.st.16x32bx2.x8(ptr addrspace(6) {{%[0-9]+}}, i64 2, <8 x i32> {{%[0-9]+}}, i1 false)
  nvvm.tcgen05.st %tmemAddr, %stv8, %offset { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>, num=8:i32 } : vector<8xi32>

// CHECK: call void @llvm.nvvm.tcgen05.st.16x32bx2.x16(ptr addrspace(6) {{%[0-9]+}}, i64 2, <16 x i32> {{%[0-9]+}}, i1 false)
  nvvm.tcgen05.st %tmemAddr, %stv16, %offset { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>, num=16:i32 } : vector<16xi32>

// CHECK: call void @llvm.nvvm.tcgen05.st.16x32bx2.x32(ptr addrspace(6) {{%[0-9]+}}, i64 2, <32 x i32> {{%[0-9]+}}, i1 false)
  nvvm.tcgen05.st %tmemAddr, %stv32, %offset { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>, num=32:i32 } : vector<32xi32>

// CHECK: call void @llvm.nvvm.tcgen05.st.16x32bx2.x64(ptr addrspace(6) {{%[0-9]+}}, i64 2, <64 x i32> {{%[0-9]+}}, i1 false)
  nvvm.tcgen05.st %tmemAddr, %stv64, %offset { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>, num=64:i32 } : vector<64xi32>

// CHECK: call void @llvm.nvvm.tcgen05.st.16x32bx2.x128(ptr addrspace(6) {{%[0-9]+}}, i64 2, <128 x i32> {{%[0-9]+}}, i1 false)
  nvvm.tcgen05.st %tmemAddr, %stv128, %offset { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>, num=128:i32 } : vector<128xi32>

  llvm.return
}

// CHECK-LABEL: @nvvm_tcgen05_ld_16x32bx2_pack
llvm.func @nvvm_tcgen05_ld_16x32bx2_pack(
  %tmemAddr : !llvm.ptr<6>,
  %stv1     : i32,
  %stv2     : vector<2xi32>,
  %stv4     : vector<4xi32>,
  %stv8     : vector<8xi32>,
  %stv16    : vector<16xi32>,
  %stv32    : vector<32xi32>,
  %stv64    : vector<64xi32>,
  %stv128   : vector<128xi32>) {

  %offset = llvm.mlir.constant(2:i64) : i64

// CHECK: call void @llvm.nvvm.tcgen05.st.16x32bx2.x1(ptr addrspace(6) {{%[0-9]+}}, i64 2, i32 {{%[0-9]+}}, i1 true)
  nvvm.tcgen05.st %tmemAddr, %stv1, %offset unpack { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>, num=1:i32 } : i32

// CHECK: call void @llvm.nvvm.tcgen05.st.16x32bx2.x2(ptr addrspace(6) {{%[0-9]+}}, i64 2, <2 x i32> {{%[0-9]+}}, i1 true)
  nvvm.tcgen05.st %tmemAddr, %stv2, %offset unpack { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>, num=2:i32 } : vector<2xi32>

// CHECK: call void @llvm.nvvm.tcgen05.st.16x32bx2.x4(ptr addrspace(6) {{%[0-9]+}}, i64 2, <4 x i32> {{%[0-9]+}}, i1 true)
  nvvm.tcgen05.st %tmemAddr, %stv4, %offset unpack { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>, num=4:i32 } : vector<4xi32>

// CHECK: call void @llvm.nvvm.tcgen05.st.16x32bx2.x8(ptr addrspace(6) {{%[0-9]+}}, i64 2, <8 x i32> {{%[0-9]+}}, i1 true)
  nvvm.tcgen05.st %tmemAddr, %stv8, %offset unpack { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>, num=8:i32 } : vector<8xi32>

// CHECK: call void @llvm.nvvm.tcgen05.st.16x32bx2.x16(ptr addrspace(6) {{%[0-9]+}}, i64 2, <16 x i32> {{%[0-9]+}}, i1 true)
  nvvm.tcgen05.st %tmemAddr, %stv16, %offset unpack { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>, num=16:i32 } : vector<16xi32>

// CHECK: call void @llvm.nvvm.tcgen05.st.16x32bx2.x32(ptr addrspace(6) {{%[0-9]+}}, i64 2, <32 x i32> {{%[0-9]+}}, i1 true)
  nvvm.tcgen05.st %tmemAddr, %stv32, %offset unpack { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>, num=32:i32 } : vector<32xi32>

// CHECK: call void @llvm.nvvm.tcgen05.st.16x32bx2.x64(ptr addrspace(6) {{%[0-9]+}}, i64 2, <64 x i32> {{%[0-9]+}}, i1 true)
  nvvm.tcgen05.st %tmemAddr, %stv64, %offset unpack { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>, num=64:i32 } : vector<64xi32>

// CHECK: call void @llvm.nvvm.tcgen05.st.16x32bx2.x128(ptr addrspace(6) {{%[0-9]+}}, i64 2, <128 x i32> {{%[0-9]+}}, i1 true)
  nvvm.tcgen05.st %tmemAddr, %stv128, %offset unpack { shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>, num=128:i32 } : vector<128xi32>

  llvm.return
}
