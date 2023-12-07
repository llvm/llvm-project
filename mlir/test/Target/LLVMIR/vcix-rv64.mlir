// RUN: mlir-translate --mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: define void @unary_ro_e8mf8(i64 %0) {
// CHECK:   call void @llvm.riscv.sf.vc.i.se.e8mf8.i64.i64.i64(i64 3, i64 31, i64 30, i64 0, i64 %0)
// CHECK:   ret void
// CHECK: }
llvm.func @unary_ro_e8mf8(%arg0: i64) {
  %0 = llvm.mlir.constant(0 : i64) : i64
  "vcix.intrin.unary.ro"(%0, %arg0) <{opcode = 3 : i64, rd = 30 : i64, rs2 = 31 : i64, sew_lmul = 0 : i32}> : (i64, i64) -> ()
  llvm.return
}

// CHECK-LABEL: define void @unary_ro_e8mf4(i64 %0) {
// CHECK:   call void @llvm.riscv.sf.vc.i.se.e8mf4.i64.i64.i64(i64 3, i64 31, i64 30, i64 0, i64 %0)
// CHECK:   ret void
// CHECK: }
llvm.func @unary_ro_e8mf4(%arg0: i64) {
  %0 = llvm.mlir.constant(0 : i64) : i64
  "vcix.intrin.unary.ro"(%0, %arg0) <{opcode = 3 : i64, rd = 30 : i64, rs2 = 31 : i64, sew_lmul = 1 : i32}> : (i64, i64) -> ()
  llvm.return
}

// CHECK-LABEL: define void @unary_ro_e8mf2(i64 %0) {
// CHECK:   call void @llvm.riscv.sf.vc.i.se.e8mf2.i64.i64.i64(i64 3, i64 31, i64 30, i64 0, i64 %0)
// CHECK:   ret void
// CHECK: }
llvm.func @unary_ro_e8mf2(%arg0: i64) {
  %0 = llvm.mlir.constant(0 : i64) : i64
  "vcix.intrin.unary.ro"(%0, %arg0) <{opcode = 3 : i64, rd = 30 : i64, rs2 = 31 : i64, sew_lmul = 2 : i32}> : (i64, i64) -> ()
  llvm.return
}

// CHECK-LABEL: define void @unary_ro_e8m1(i64 %0) {
// CHECK:   call void @llvm.riscv.sf.vc.i.se.e8m1.i64.i64.i64(i64 3, i64 31, i64 30, i64 0, i64 %0)
// CHECK:   ret void
// CHECK: }
llvm.func @unary_ro_e8m1(%arg0: i64) {
  %0 = llvm.mlir.constant(0 : i64) : i64
  "vcix.intrin.unary.ro"(%0, %arg0) <{opcode = 3 : i64, rd = 30 : i64, rs2 = 31 : i64, sew_lmul = 3 : i32}> : (i64, i64) -> ()
  llvm.return
}

// CHECK-LABEL: define void @unary_ro_e8m2(i64 %0) {
// CHECK:   call void @llvm.riscv.sf.vc.i.se.e8m2.i64.i64.i64(i64 3, i64 31, i64 30, i64 0, i64 %0)
// CHECK:   ret void
// CHECK: }
llvm.func @unary_ro_e8m2(%arg0: i64) {
  %0 = llvm.mlir.constant(0 : i64) : i64
  "vcix.intrin.unary.ro"(%0, %arg0) <{opcode = 3 : i64, rd = 30 : i64, rs2 = 31 : i64, sew_lmul = 4 : i32}> : (i64, i64) -> ()
  llvm.return
}

// CHECK-LABEL: define void @unary_ro_e8m4(i64 %0) {
// CHECK:   call void @llvm.riscv.sf.vc.i.se.e8m4.i64.i64.i64(i64 3, i64 31, i64 30, i64 0, i64 %0)
// CHECK:   ret void
// CHECK: }
llvm.func @unary_ro_e8m4(%arg0: i64) {
  %0 = llvm.mlir.constant(0 : i64) : i64
  "vcix.intrin.unary.ro"(%0, %arg0) <{opcode = 3 : i64, rd = 30 : i64, rs2 = 31 : i64, sew_lmul = 5 : i32}> : (i64, i64) -> ()
  llvm.return
}

// CHECK-LABEL: define void @unary_ro_e8m8(i64 %0) {
// CHECK:   call void @llvm.riscv.sf.vc.i.se.e8m8.i64.i64.i64(i64 3, i64 31, i64 30, i64 0, i64 %0)
// CHECK:   ret void
// CHECK: }
llvm.func @unary_ro_e8m8(%arg0: i64) {
  %0 = llvm.mlir.constant(0 : i64) : i64
  "vcix.intrin.unary.ro"(%0, %arg0) <{opcode = 3 : i64, rd = 30 : i64, rs2 = 31 : i64, sew_lmul = 6 : i32}> : (i64, i64) -> ()
  llvm.return
}

// CHECK-LABEL: define <vscale x 1 x i8> @unary_e8mf8(i64 %0) {
// CHECK:   %2 = call <vscale x 1 x i8> @llvm.riscv.sf.vc.v.x.nxv1i8.i64.i64.i64(i64 3, i64 31, i64 0, i64 %0)
// CHECK:   ret <vscale x 1 x i8> %2
// CHECK: }
llvm.func @unary_e8mf8(%arg0: i64) -> vector<[1]xi8> {
  %0 = llvm.mlir.constant(0 : i64) : i64
  %1 = "vcix.intrin.unary"(%0, %arg0) <{opcode = 3 : i64, rs2 = 31 : i64}> : (i64, i64) -> vector<[1]xi8>
  llvm.return %1 : vector<[1]xi8>
}

// CHECK-LABEL: define <vscale x 2 x i8> @unary_e8mf4(i64 %0) {
// CHECK:   %2 = call <vscale x 2 x i8> @llvm.riscv.sf.vc.v.x.nxv2i8.i64.i64.i64(i64 3, i64 31, i64 0, i64 %0)
// CHECK:   ret <vscale x 2 x i8> %2
// CHECK: }
llvm.func @unary_e8mf4(%arg0: i64) -> vector<[2]xi8> {
  %0 = llvm.mlir.constant(0 : i64) : i64
  %1 = "vcix.intrin.unary"(%0, %arg0) <{opcode = 3 : i64, rs2 = 31 : i64}> : (i64, i64) -> vector<[2]xi8>
  llvm.return %1 : vector<[2]xi8>
}

// CHECK-LABEL: define <vscale x 4 x i8> @unary_e8mf2(i64 %0) {
// CHECK:   %2 = call <vscale x 4 x i8> @llvm.riscv.sf.vc.v.x.nxv4i8.i64.i64.i64(i64 3, i64 31, i64 0, i64 %0)
// CHECK:   ret <vscale x 4 x i8> %2
// CHECK: }
llvm.func @unary_e8mf2(%arg0: i64) -> vector<[4]xi8> {
  %0 = llvm.mlir.constant(0 : i64) : i64
  %1 = "vcix.intrin.unary"(%0, %arg0) <{opcode = 3 : i64, rs2 = 31 : i64}> : (i64, i64) -> vector<[4]xi8>
  llvm.return %1 : vector<[4]xi8>
}

// CHECK-LABEL: define <vscale x 8 x i8> @unary_e8m1(i64 %0) {
// CHECK:   %2 = call <vscale x 8 x i8> @llvm.riscv.sf.vc.v.x.nxv8i8.i64.i64.i64(i64 3, i64 31, i64 0, i64 %0)
// CHECK:   ret <vscale x 8 x i8> %2
// CHECK: }
llvm.func @unary_e8m1(%arg0: i64) -> vector<[8]xi8> {
  %0 = llvm.mlir.constant(0 : i64) : i64
  %1 = "vcix.intrin.unary"(%0, %arg0) <{opcode = 3 : i64, rs2 = 31 : i64}> : (i64, i64) -> vector<[8]xi8>
  llvm.return %1 : vector<[8]xi8>
}

// CHECK-LABEL: define <vscale x 16 x i8> @unary_e8m2(i64 %0) {
// CHECK:   %2 = call <vscale x 16 x i8> @llvm.riscv.sf.vc.v.x.nxv16i8.i64.i64.i64(i64 3, i64 31, i64 0, i64 %0)
// CHECK:   ret <vscale x 16 x i8> %2
// CHECK: }
llvm.func @unary_e8m2(%arg0: i64) -> vector<[16]xi8> {
  %0 = llvm.mlir.constant(0 : i64) : i64
  %1 = "vcix.intrin.unary"(%0, %arg0) <{opcode = 3 : i64, rs2 = 31 : i64}> : (i64, i64) -> vector<[16]xi8>
  llvm.return %1 : vector<[16]xi8>
}

// CHECK-LABEL: define <vscale x 32 x i8> @unary_e8m4(i64 %0) {
// CHECK:   %2 = call <vscale x 32 x i8> @llvm.riscv.sf.vc.v.x.nxv32i8.i64.i64.i64(i64 3, i64 31, i64 0, i64 %0)
// CHECK:   ret <vscale x 32 x i8> %2
// CHECK: }
llvm.func @unary_e8m4(%arg0: i64) -> vector<[32]xi8> {
  %0 = llvm.mlir.constant(0 : i64) : i64
  %1 = "vcix.intrin.unary"(%0, %arg0) <{opcode = 3 : i64, rs2 = 31 : i64}> : (i64, i64) -> vector<[32]xi8>
  llvm.return %1 : vector<[32]xi8>
}

// CHECK-LABEL: define <vscale x 64 x i8> @unary_e8m8(i64 %0) {
// CHECK:   %2 = call <vscale x 64 x i8> @llvm.riscv.sf.vc.v.x.nxv64i8.i64.i64.i64(i64 3, i64 31, i64 0, i64 %0)
// CHECK:   ret <vscale x 64 x i8> %2
// CHECK: }
llvm.func @unary_e8m8(%arg0: i64) -> vector<[64]xi8> {
  %0 = llvm.mlir.constant(0 : i64) : i64
  %1 = "vcix.intrin.unary"(%0, %arg0) <{opcode = 3 : i64, rs2 = 31 : i64}> : (i64, i64) -> vector<[64]xi8>
  llvm.return %1 : vector<[64]xi8>
}

// CHECK-LABEL: define void @unary_ro_e16mf4(i64 %0) {
// CHECK:   call void @llvm.riscv.sf.vc.i.se.e16mf4.i64.i64.i64(i64 3, i64 31, i64 30, i64 0, i64 %0)
// CHECK:   ret void
// CHECK: }
llvm.func @unary_ro_e16mf4(%arg0: i64) {
  %0 = llvm.mlir.constant(0 : i64) : i64
  "vcix.intrin.unary.ro"(%0, %arg0) <{opcode = 3 : i64, rd = 30 : i64, rs2 = 31 : i64, sew_lmul = 7 : i32}> : (i64, i64) -> ()
  llvm.return
}

// CHECK-LABEL: define void @unary_ro_e16mf2(i64 %0) {
// CHECK:   call void @llvm.riscv.sf.vc.i.se.e16mf2.i64.i64.i64(i64 3, i64 31, i64 30, i64 0, i64 %0)
// CHECK:   ret void
// CHECK: }
llvm.func @unary_ro_e16mf2(%arg0: i64) {
  %0 = llvm.mlir.constant(0 : i64) : i64
  "vcix.intrin.unary.ro"(%0, %arg0) <{opcode = 3 : i64, rd = 30 : i64, rs2 = 31 : i64, sew_lmul = 8 : i32}> : (i64, i64) -> ()
  llvm.return
}

// CHECK-LABEL: define void @unary_ro_e16m1(i64 %0) {
// CHECK:   call void @llvm.riscv.sf.vc.i.se.e16m1.i64.i64.i64(i64 3, i64 31, i64 30, i64 0, i64 %0)
// CHECK:   ret void
// CHECK: }
llvm.func @unary_ro_e16m1(%arg0: i64) {
  %0 = llvm.mlir.constant(0 : i64) : i64
  "vcix.intrin.unary.ro"(%0, %arg0) <{opcode = 3 : i64, rd = 30 : i64, rs2 = 31 : i64, sew_lmul = 9 : i32}> : (i64, i64) -> ()
  llvm.return
}

// CHECK-LABEL: define void @unary_ro_e16m2(i64 %0) {
// CHECK:   call void @llvm.riscv.sf.vc.i.se.e16m2.i64.i64.i64(i64 3, i64 31, i64 30, i64 0, i64 %0)
// CHECK:   ret void
// CHECK: }
llvm.func @unary_ro_e16m2(%arg0: i64) {
  %0 = llvm.mlir.constant(0 : i64) : i64
  "vcix.intrin.unary.ro"(%0, %arg0) <{opcode = 3 : i64, rd = 30 : i64, rs2 = 31 : i64, sew_lmul = 10 : i32}> : (i64, i64) -> ()
  llvm.return
}

// CHECK-LABEL: define void @unary_ro_e16m4(i64 %0) {
// CHECK:   call void @llvm.riscv.sf.vc.i.se.e16m4.i64.i64.i64(i64 3, i64 31, i64 30, i64 0, i64 %0)
// CHECK:   ret void
// CHECK: }
llvm.func @unary_ro_e16m4(%arg0: i64) {
  %0 = llvm.mlir.constant(0 : i64) : i64
  "vcix.intrin.unary.ro"(%0, %arg0) <{opcode = 3 : i64, rd = 30 : i64, rs2 = 31 : i64, sew_lmul = 11 : i32}> : (i64, i64) -> ()
  llvm.return
}

// CHECK-LABEL: define void @unary_ro_e16m8(i64 %0) {
// CHECK:   call void @llvm.riscv.sf.vc.i.se.e16m8.i64.i64.i64(i64 3, i64 31, i64 30, i64 0, i64 %0)
// CHECK:   ret void
// CHECK: }
llvm.func @unary_ro_e16m8(%arg0: i64) {
  %0 = llvm.mlir.constant(0 : i64) : i64
  "vcix.intrin.unary.ro"(%0, %arg0) <{opcode = 3 : i64, rd = 30 : i64, rs2 = 31 : i64, sew_lmul = 12 : i32}> : (i64, i64) -> ()
  llvm.return
}

// CHECK-LABEL: define <vscale x 1 x i16> @unary_e16mf4(i64 %0) {
// CHECK:   %2 = call <vscale x 1 x i16> @llvm.riscv.sf.vc.v.x.nxv1i16.i64.i64.i64(i64 3, i64 31, i64 0, i64 %0)
// CHECK:   ret <vscale x 1 x i16> %2
// CHECK: }
llvm.func @unary_e16mf4(%arg0: i64) -> vector<[1]xi16> {
  %0 = llvm.mlir.constant(0 : i64) : i64
  %1 = "vcix.intrin.unary"(%0, %arg0) <{opcode = 3 : i64, rs2 = 31 : i64}> : (i64, i64) -> vector<[1]xi16>
  llvm.return %1 : vector<[1]xi16>
}

// CHECK-LABEL: define <vscale x 2 x i16> @unary_e16mf2(i64 %0) {
// CHECK:   %2 = call <vscale x 2 x i16> @llvm.riscv.sf.vc.v.x.nxv2i16.i64.i64.i64(i64 3, i64 31, i64 0, i64 %0)
// CHECK:   ret <vscale x 2 x i16> %2
// CHECK: }
llvm.func @unary_e16mf2(%arg0: i64) -> vector<[2]xi16> {
  %0 = llvm.mlir.constant(0 : i64) : i64
  %1 = "vcix.intrin.unary"(%0, %arg0) <{opcode = 3 : i64, rs2 = 31 : i64}> : (i64, i64) -> vector<[2]xi16>
  llvm.return %1 : vector<[2]xi16>
}

// CHECK-LABEL: define <vscale x 4 x i16> @unary_e16m1(i64 %0) {
// CHECK:   %2 = call <vscale x 4 x i16> @llvm.riscv.sf.vc.v.x.nxv4i16.i64.i64.i64(i64 3, i64 31, i64 0, i64 %0)
// CHECK:   ret <vscale x 4 x i16> %2
// CHECK: }
llvm.func @unary_e16m1(%arg0: i64) -> vector<[4]xi16> {
  %0 = llvm.mlir.constant(0 : i64) : i64
  %1 = "vcix.intrin.unary"(%0, %arg0) <{opcode = 3 : i64, rs2 = 31 : i64}> : (i64, i64) -> vector<[4]xi16>
  llvm.return %1 : vector<[4]xi16>
}

// CHECK-LABEL: define <vscale x 8 x i16> @unary_e16m2(i64 %0) {
// CHECK:   %2 = call <vscale x 8 x i16> @llvm.riscv.sf.vc.v.x.nxv8i16.i64.i64.i64(i64 3, i64 31, i64 0, i64 %0)
// CHECK:   ret <vscale x 8 x i16> %2
// CHECK: }
llvm.func @unary_e16m2(%arg0: i64) -> vector<[8]xi16> {
  %0 = llvm.mlir.constant(0 : i64) : i64
  %1 = "vcix.intrin.unary"(%0, %arg0) <{opcode = 3 : i64, rs2 = 31 : i64}> : (i64, i64) -> vector<[8]xi16>
  llvm.return %1 : vector<[8]xi16>
}

// CHECK-LABEL: define <vscale x 16 x i16> @unary_e16m4(i64 %0) {
// CHECK:   %2 = call <vscale x 16 x i16> @llvm.riscv.sf.vc.v.x.nxv16i16.i64.i64.i64(i64 3, i64 31, i64 0, i64 %0)
// CHECK:   ret <vscale x 16 x i16> %2
// CHECK: }
llvm.func @unary_e16m4(%arg0: i64) -> vector<[16]xi16> {
  %0 = llvm.mlir.constant(0 : i64) : i64
  %1 = "vcix.intrin.unary"(%0, %arg0) <{opcode = 3 : i64, rs2 = 31 : i64}> : (i64, i64) -> vector<[16]xi16>
  llvm.return %1 : vector<[16]xi16>
}

// CHECK-LABEL: define <vscale x 32 x i16> @unary_e16m8(i64 %0) {
// CHECK:   %2 = call <vscale x 32 x i16> @llvm.riscv.sf.vc.v.x.nxv32i16.i64.i64.i64(i64 3, i64 31, i64 0, i64 %0)
// CHECK:   ret <vscale x 32 x i16> %2
// CHECK: }
llvm.func @unary_e16m8(%arg0: i64) -> vector<[32]xi16> {
  %0 = llvm.mlir.constant(0 : i64) : i64
  %1 = "vcix.intrin.unary"(%0, %arg0) <{opcode = 3 : i64, rs2 = 31 : i64}> : (i64, i64) -> vector<[32]xi16>
  llvm.return %1 : vector<[32]xi16>
}

// CHECK-LABEL: define void @unary_ro_e32mf2(i64 %0) {
// CHECK:   call void @llvm.riscv.sf.vc.i.se.e32mf2.i64.i64.i64(i64 3, i64 31, i64 30, i64 0, i64 %0)
// CHECK:   ret void
// CHECK: }
llvm.func @unary_ro_e32mf2(%arg0: i64) {
  %0 = llvm.mlir.constant(0 : i64) : i64
  "vcix.intrin.unary.ro"(%0, %arg0) <{opcode = 3 : i64, rd = 30 : i64, rs2 = 31 : i64, sew_lmul = 13 : i32}> : (i64, i64) -> ()
  llvm.return
}

// CHECK-LABEL: define void @unary_ro_e32m1(i64 %0) {
// CHECK:   call void @llvm.riscv.sf.vc.i.se.e32m1.i64.i64.i64(i64 3, i64 31, i64 30, i64 0, i64 %0)
// CHECK:   ret void
// CHECK: }
llvm.func @unary_ro_e32m1(%arg0: i64) {
  %0 = llvm.mlir.constant(0 : i64) : i64
  "vcix.intrin.unary.ro"(%0, %arg0) <{opcode = 3 : i64, rd = 30 : i64, rs2 = 31 : i64, sew_lmul = 14 : i32}> : (i64, i64) -> ()
  llvm.return
}

// CHECK-LABEL: define void @unary_ro_e32m2(i64 %0) {
// CHECK:   call void @llvm.riscv.sf.vc.i.se.e32m2.i64.i64.i64(i64 3, i64 31, i64 30, i64 0, i64 %0)
// CHECK:   ret void
// CHECK: }
llvm.func @unary_ro_e32m2(%arg0: i64) {
  %0 = llvm.mlir.constant(0 : i64) : i64
  "vcix.intrin.unary.ro"(%0, %arg0) <{opcode = 3 : i64, rd = 30 : i64, rs2 = 31 : i64, sew_lmul = 15 : i32}> : (i64, i64) -> ()
  llvm.return
}

// CHECK-LABEL: define void @unary_ro_e32m4(i64 %0) {
// CHECK:   call void @llvm.riscv.sf.vc.i.se.e32m4.i64.i64.i64(i64 3, i64 31, i64 30, i64 0, i64 %0)
// CHECK:   ret void
// CHECK: }
llvm.func @unary_ro_e32m4(%arg0: i64) {
  %0 = llvm.mlir.constant(0 : i64) : i64
  "vcix.intrin.unary.ro"(%0, %arg0) <{opcode = 3 : i64, rd = 30 : i64, rs2 = 31 : i64, sew_lmul = 16 : i32}> : (i64, i64) -> ()
  llvm.return
}

// CHECK-LABEL: define void @unary_ro_e32m8(i64 %0) {
// CHECK:   call void @llvm.riscv.sf.vc.i.se.e32m8.i64.i64.i64(i64 3, i64 31, i64 30, i64 0, i64 %0)
// CHECK:   ret void
// CHECK: }
llvm.func @unary_ro_e32m8(%arg0: i64) {
  %0 = llvm.mlir.constant(0 : i64) : i64
  "vcix.intrin.unary.ro"(%0, %arg0) <{opcode = 3 : i64, rd = 30 : i64, rs2 = 31 : i64, sew_lmul = 17 : i32}> : (i64, i64) -> ()
  llvm.return
}

// CHECK-LABEL: define <vscale x 1 x i32> @unary_e32mf2(i64 %0) {
// CHECK:   %2 = call <vscale x 1 x i32> @llvm.riscv.sf.vc.v.x.nxv1i32.i64.i64.i64(i64 3, i64 31, i64 0, i64 %0)
// CHECK:   ret <vscale x 1 x i32> %2
// CHECK: }
llvm.func @unary_e32mf2(%arg0: i64) -> vector<[1]xi32> {
  %0 = llvm.mlir.constant(0 : i64) : i64
  %1 = "vcix.intrin.unary"(%0, %arg0) <{opcode = 3 : i64, rs2 = 31 : i64}> : (i64, i64) -> vector<[1]xi32>
  llvm.return %1 : vector<[1]xi32>
}

// CHECK-LABEL: define <vscale x 2 x i32> @unary_e32m1(i64 %0) {
// CHECK:   %2 = call <vscale x 2 x i32> @llvm.riscv.sf.vc.v.x.nxv2i32.i64.i64.i64(i64 3, i64 31, i64 0, i64 %0)
// CHECK:   ret <vscale x 2 x i32> %2
// CHECK: }
llvm.func @unary_e32m1(%arg0: i64) -> vector<[2]xi32> {
  %0 = llvm.mlir.constant(0 : i64) : i64
  %1 = "vcix.intrin.unary"(%0, %arg0) <{opcode = 3 : i64, rs2 = 31 : i64}> : (i64, i64) -> vector<[2]xi32>
  llvm.return %1 : vector<[2]xi32>
}

// CHECK-LABEL: define <vscale x 4 x i32> @unary_e32m2(i64 %0) {
// CHECK:   %2 = call <vscale x 4 x i32> @llvm.riscv.sf.vc.v.x.nxv4i32.i64.i64.i64(i64 3, i64 31, i64 0, i64 %0)
// CHECK:   ret <vscale x 4 x i32> %2
// CHECK: }
llvm.func @unary_e32m2(%arg0: i64) -> vector<[4]xi32> {
  %0 = llvm.mlir.constant(0 : i64) : i64
  %1 = "vcix.intrin.unary"(%0, %arg0) <{opcode = 3 : i64, rs2 = 31 : i64}> : (i64, i64) -> vector<[4]xi32>
  llvm.return %1 : vector<[4]xi32>
}

// CHECK-LABEL: define <vscale x 8 x i32> @unary_e32m4(i64 %0) {
// CHECK:   %2 = call <vscale x 8 x i32> @llvm.riscv.sf.vc.v.x.nxv8i32.i64.i64.i64(i64 3, i64 31, i64 0, i64 %0)
// CHECK:   ret <vscale x 8 x i32> %2
// CHECK: }
llvm.func @unary_e32m4(%arg0: i64) -> vector<[8]xi32> {
  %0 = llvm.mlir.constant(0 : i64) : i64
  %1 = "vcix.intrin.unary"(%0, %arg0) <{opcode = 3 : i64, rs2 = 31 : i64}> : (i64, i64) -> vector<[8]xi32>
  llvm.return %1 : vector<[8]xi32>
}

// CHECK-LABEL: define <vscale x 16 x i32> @unary_e32m8(i64 %0) {
// CHECK:   %2 = call <vscale x 16 x i32> @llvm.riscv.sf.vc.v.x.nxv16i32.i64.i64.i64(i64 3, i64 31, i64 0, i64 %0)
// CHECK:   ret <vscale x 16 x i32> %2
// CHECK: }
llvm.func @unary_e32m8(%arg0: i64) -> vector<[16]xi32> {
  %0 = llvm.mlir.constant(0 : i64) : i64
  %1 = "vcix.intrin.unary"(%0, %arg0) <{opcode = 3 : i64, rs2 = 31 : i64}> : (i64, i64) -> vector<[16]xi32>
  llvm.return %1 : vector<[16]xi32>
}

// CHECK-LABEL: define void @unary_ro_e64m1(i64 %0) {
// CHECK:   call void @llvm.riscv.sf.vc.i.se.e64m1.i64.i64.i64(i64 3, i64 31, i64 30, i64 0, i64 %0)
// CHECK:   ret void
// CHECK: }
llvm.func @unary_ro_e64m1(%arg0: i64) {
  %0 = llvm.mlir.constant(0 : i64) : i64
  "vcix.intrin.unary.ro"(%0, %arg0) <{opcode = 3 : i64, rd = 30 : i64, rs2 = 31 : i64, sew_lmul = 18 : i32}> : (i64, i64) -> ()
  llvm.return
}

// CHECK-LABEL: define void @unary_ro_e64m2(i64 %0) {
// CHECK:   call void @llvm.riscv.sf.vc.i.se.e64m2.i64.i64.i64(i64 3, i64 31, i64 30, i64 0, i64 %0)
// CHECK:   ret void
// CHECK: }
llvm.func @unary_ro_e64m2(%arg0: i64) {
  %0 = llvm.mlir.constant(0 : i64) : i64
  "vcix.intrin.unary.ro"(%0, %arg0) <{opcode = 3 : i64, rd = 30 : i64, rs2 = 31 : i64, sew_lmul = 19 : i32}> : (i64, i64) -> ()
  llvm.return
}

// CHECK-LABEL: define void @unary_ro_e64m4(i64 %0) {
// CHECK:   call void @llvm.riscv.sf.vc.i.se.e64m4.i64.i64.i64(i64 3, i64 31, i64 30, i64 0, i64 %0)
// CHECK:   ret void
// CHECK: }
llvm.func @unary_ro_e64m4(%arg0: i64) {
  %0 = llvm.mlir.constant(0 : i64) : i64
  "vcix.intrin.unary.ro"(%0, %arg0) <{opcode = 3 : i64, rd = 30 : i64, rs2 = 31 : i64, sew_lmul = 20 : i32}> : (i64, i64) -> ()
  llvm.return
}

// CHECK-LABEL: define void @unary_ro_e64m8(i64 %0) {
// CHECK:   call void @llvm.riscv.sf.vc.i.se.e64m8.i64.i64.i64(i64 3, i64 31, i64 30, i64 0, i64 %0)
// CHECK:   ret void
// CHECK: }
llvm.func @unary_ro_e64m8(%arg0: i64) {
  %0 = llvm.mlir.constant(0 : i64) : i64
  "vcix.intrin.unary.ro"(%0, %arg0) <{opcode = 3 : i64, rd = 30 : i64, rs2 = 31 : i64, sew_lmul = 21 : i32}> : (i64, i64) -> ()
  llvm.return
}

// CHECK-LABEL: define <vscale x 1 x i32> @unary_e64m1(i64 %0) {
// CHECK:   %2 = call <vscale x 1 x i32> @llvm.riscv.sf.vc.v.x.nxv1i32.i64.i64.i64(i64 3, i64 31, i64 0, i64 %0)
// CHECK:   ret <vscale x 1 x i32> %2
// CHECK: }
llvm.func @unary_e64m1(%arg0: i64) -> vector<[1]xi32> {
  %0 = llvm.mlir.constant(0 : i64) : i64
  %1 = "vcix.intrin.unary"(%0, %arg0) <{opcode = 3 : i64, rs2 = 31 : i64}> : (i64, i64) -> vector<[1]xi32>
  llvm.return %1 : vector<[1]xi32>
}

// CHECK-LABEL: define <vscale x 2 x i32> @unary_e64m2(i64 %0) {
// CHECK:   %2 = call <vscale x 2 x i32> @llvm.riscv.sf.vc.v.x.nxv2i32.i64.i64.i64(i64 3, i64 31, i64 0, i64 %0)
// CHECK:   ret <vscale x 2 x i32> %2
// CHECK: }
llvm.func @unary_e64m2(%arg0: i64) -> vector<[2]xi32> {
  %0 = llvm.mlir.constant(0 : i64) : i64
  %1 = "vcix.intrin.unary"(%0, %arg0) <{opcode = 3 : i64, rs2 = 31 : i64}> : (i64, i64) -> vector<[2]xi32>
  llvm.return %1 : vector<[2]xi32>
}

// CHECK-LABEL: define <vscale x 4 x i32> @unary_e64m4(i64 %0) {
// CHECK:   %2 = call <vscale x 4 x i32> @llvm.riscv.sf.vc.v.x.nxv4i32.i64.i64.i64(i64 3, i64 31, i64 0, i64 %0)
// CHECK:   ret <vscale x 4 x i32> %2
// CHECK: }
llvm.func @unary_e64m4(%arg0: i64) -> vector<[4]xi32> {
  %0 = llvm.mlir.constant(0 : i64) : i64
  %1 = "vcix.intrin.unary"(%0, %arg0) <{opcode = 3 : i64, rs2 = 31 : i64}> : (i64, i64) -> vector<[4]xi32>
  llvm.return %1 : vector<[4]xi32>
}

// CHECK-LABEL: define <vscale x 8 x i32> @unary_e64m8(i64 %0) {
// CHECK:   %2 = call <vscale x 8 x i32> @llvm.riscv.sf.vc.v.x.nxv8i32.i64.i64.i64(i64 3, i64 31, i64 0, i64 %0)
// CHECK:   ret <vscale x 8 x i32> %2
// CHECK: }
llvm.func @unary_e64m8(%arg0: i64) -> vector<[8]xi32> {
  %0 = llvm.mlir.constant(0 : i64) : i64
  %1 = "vcix.intrin.unary"(%0, %arg0) <{opcode = 3 : i64, rs2 = 31 : i64}> : (i64, i64) -> vector<[8]xi32>
  llvm.return %1 : vector<[8]xi32>
}

// CHECK-LABEL: define void @binary_vv_ro(<vscale x 4 x float> %0, <vscale x 4 x float> %1, i64 %2) {
// CHECK:   call void @llvm.riscv.sf.vc.vv.se.i64.nxv4f32.nxv4f32.i64(i64 3, i64 30, <vscale x 4 x float> %1, <vscale x 4 x float> %0, i64 %2)
// CHECK:   ret void
// CHECK: }
llvm.func @binary_vv_ro(%arg0: vector<[4]xf32>, %arg1: vector<[4]xf32>, %arg2: i64) {
  "vcix.intrin.binary.ro"(%arg0, %arg1, %arg2) <{opcode = 3 : i64, rd = 30 : i64}> : (vector<[4]xf32>, vector<[4]xf32>, i64) -> ()
  llvm.return
}

// CHECK-LABEL: define <vscale x 4 x float> @binary_vv(<vscale x 4 x float> %0, <vscale x 4 x float> %1, i64 %2) {
// CHECK:   %4 = call <vscale x 4 x float> @llvm.riscv.sf.vc.v.vv.se.nxv4f32.i64.nxv4f32.nxv4f32.i64(i64 3, <vscale x 4 x float> %1, <vscale x 4 x float> %0, i64 %2)
// CHECK:   ret <vscale x 4 x float> %4
// CHECK: }
llvm.func @binary_vv(%arg0: vector<[4]xf32>, %arg1: vector<[4]xf32>, %arg2: i64) -> vector<[4]xf32> {
  %0 = "vcix.intrin.binary"(%arg0, %arg1, %arg2) <{opcode = 3 : i64}> : (vector<[4]xf32>, vector<[4]xf32>, i64) -> vector<[4]xf32>
  llvm.return %0 : vector<[4]xf32>
}

// CHECK-LABEL: define void @binary_xv_ro(i64 %0, <vscale x 4 x float> %1, i64 %2) {
// CHECK:   call void @llvm.riscv.sf.vc.xv.se.i64.nxv4f32.i64.i64(i64 3, i64 30, <vscale x 4 x float> %1, i64 %0, i64 %2)
// CHECK:   ret void
// CHECK: }
llvm.func @binary_xv_ro(%arg0: i64, %arg1: vector<[4]xf32>, %arg2: i64) {
  "vcix.intrin.binary.ro"(%arg0, %arg1, %arg2) <{opcode = 3 : i64, rd = 30 : i64}> : (i64, vector<[4]xf32>, i64) -> ()
  llvm.return
}

// CHECK-LABEL: define <vscale x 4 x float> @binary_xv(i64 %0, <vscale x 4 x float> %1, i64 %2) {
// CHECK:   %4 = call <vscale x 4 x float> @llvm.riscv.sf.vc.v.xv.se.nxv4f32.i64.nxv4f32.i64.i64(i64 3, <vscale x 4 x float> %1, i64 %0, i64 %2)
// CHECK:   ret <vscale x 4 x float> %4
// CHECK: }
llvm.func @binary_xv(%arg0: i64, %arg1: vector<[4]xf32>, %arg2: i64) -> vector<[4]xf32> {
  %0 = "vcix.intrin.binary"(%arg0, %arg1, %arg2) <{opcode = 3 : i64}> : (i64, vector<[4]xf32>, i64) -> vector<[4]xf32>
  llvm.return %0 : vector<[4]xf32>
}

// CHECK-LABEL: define void @binary_fv_ro(float %0, <vscale x 4 x float> %1, i64 %2) {
// CHECK:   call void @llvm.riscv.sf.vc.fv.se.i64.nxv4f32.f32.i64(i64 1, i64 30, <vscale x 4 x float> %1, float %0, i64 %2)
// CHECK:   ret void
// CHECK: }
llvm.func @binary_fv_ro(%arg0: f32, %arg1: vector<[4]xf32>, %arg2: i64) {
  "vcix.intrin.binary.ro"(%arg0, %arg1, %arg2) <{opcode = 1 : i64, rd = 30 : i64}> : (f32, vector<[4]xf32>, i64) -> ()
  llvm.return
}

// CHECK-LABEL: define <vscale x 4 x float> @binary_fv(float %0, <vscale x 4 x float> %1, i64 %2) {
// CHECK:   %4 = call <vscale x 4 x float> @llvm.riscv.sf.vc.v.fv.se.nxv4f32.i64.nxv4f32.f32.i64(i64 1, <vscale x 4 x float> %1, float %0, i64 %2)
// CHECK:   ret <vscale x 4 x float> %4
// CHECK: }
llvm.func @binary_fv(%arg0: f32, %arg1: vector<[4]xf32>, %arg2: i64) -> vector<[4]xf32> {
  %0 = "vcix.intrin.binary"(%arg0, %arg1, %arg2) <{opcode = 1 : i64}> : (f32, vector<[4]xf32>, i64) -> vector<[4]xf32>
  llvm.return %0 : vector<[4]xf32>
}

// CHECK-LABEL: define void @binary_iv_ro(<vscale x 4 x float> %0, i64 %1) {
// CHECK:   call void @llvm.riscv.sf.vc.xv.se.i64.nxv4f32.i64.i64(i64 3, i64 30, <vscale x 4 x float> %0, i64 1, i64 %1)
// CHECK:   ret void
// CHECK: }
llvm.func @binary_iv_ro(%arg0: vector<[4]xf32>, %arg1: i64) {
  %0 = llvm.mlir.constant(1 : i5) : i5
  %1 = llvm.zext %0 : i5 to i64
  "vcix.intrin.binary.ro"(%1, %arg0, %arg1) <{opcode = 3 : i64, rd = 30 : i64}> : (i64, vector<[4]xf32>, i64) -> ()
  llvm.return
}

// CHECK-LABEL: define <vscale x 4 x float> @binary_iv(<vscale x 4 x float> %0, i64 %1) {
// CHECK:   %3 = call <vscale x 4 x float> @llvm.riscv.sf.vc.v.xv.se.nxv4f32.i64.nxv4f32.i64.i64(i64 3, <vscale x 4 x float> %0, i64 1, i64 %1)
// CHECK:   ret <vscale x 4 x float> %3
// CHECK: }
llvm.func @binary_iv(%arg0: vector<[4]xf32>, %arg1: i64) -> vector<[4]xf32> {
  %0 = llvm.mlir.constant(1 : i5) : i5
  %1 = llvm.zext %0 : i5 to i64
  %2 = "vcix.intrin.binary"(%1, %arg0, %arg1) <{opcode = 3 : i64}> : (i64, vector<[4]xf32>, i64) -> vector<[4]xf32>
  llvm.return %2 : vector<[4]xf32>
}

// CHECK-LABEL: define void @ternary_vvv_ro(<vscale x 4 x float> %0, <vscale x 4 x float> %1, <vscale x 4 x float> %2, i64 %3) {
// CHECK:   call void @llvm.riscv.sf.vc.vvv.se.i64.nxv4f32.nxv4f32.nxv4f32.i64(i64 3, <vscale x 4 x float> %2, <vscale x 4 x float> %1, <vscale x 4 x float> %0, i64 %3)
// CHECK:   ret void
// CHECK: }
llvm.func @ternary_vvv_ro(%arg0: vector<[4]xf32>, %arg1: vector<[4]xf32>, %arg2: vector<[4]xf32>, %arg3: i64) {
  "vcix.intrin.ternary.ro"(%arg0, %arg1, %arg2, %arg3) <{opcode = 3 : i64}> : (vector<[4]xf32>, vector<[4]xf32>, vector<[4]xf32>, i64) -> ()
  llvm.return
}

// CHECK-LABEL: define <vscale x 4 x float> @ternary_vvv(<vscale x 4 x float> %0, <vscale x 4 x float> %1, <vscale x 4 x float> %2, i64 %3) {
// CHECK:   %5 = call <vscale x 4 x float> @llvm.riscv.sf.vc.v.vvv.se.nxv4f32.i64.nxv4f32.nxv4f32.nxv4f32.i64(i64 3, <vscale x 4 x float> %2, <vscale x 4 x float> %1, <vscale x 4 x float> %0, i64 %3)
// CHECK:   ret <vscale x 4 x float> %5
// CHECK: }
llvm.func @ternary_vvv(%arg0: vector<[4]xf32>, %arg1: vector<[4]xf32>, %arg2: vector<[4]xf32>, %arg3: i64) -> vector<[4]xf32> {
  %0 = "vcix.intrin.ternary"(%arg0, %arg1, %arg2, %arg3) <{opcode = 3 : i64}> : (vector<[4]xf32>, vector<[4]xf32>, vector<[4]xf32>, i64) -> vector<[4]xf32>
  llvm.return %0 : vector<[4]xf32>
}

// CHECK-LABEL: define void @ternary_xvv_ro(i64 %0, <vscale x 4 x float> %1, <vscale x 4 x float> %2, i64 %3) {
// CHECK:   call void @llvm.riscv.sf.vc.xvv.se.i64.nxv4f32.nxv4f32.i64.i64(i64 3, <vscale x 4 x float> %2, <vscale x 4 x float> %1, i64 %0, i64 %3)
// CHECK:   ret void
// CHECK: }
llvm.func @ternary_xvv_ro(%arg0: i64, %arg1: vector<[4]xf32>, %arg2: vector<[4]xf32>, %arg3: i64) {
  "vcix.intrin.ternary.ro"(%arg0, %arg1, %arg2, %arg3) <{opcode = 3 : i64}> : (i64, vector<[4]xf32>, vector<[4]xf32>, i64) -> ()
  llvm.return
}

// CHECK-LABEL: define <vscale x 4 x float> @ternary_xvv(i64 %0, <vscale x 4 x float> %1, <vscale x 4 x float> %2, i64 %3) {
// CHECK:   %5 = call <vscale x 4 x float> @llvm.riscv.sf.vc.v.xvv.se.nxv4f32.i64.nxv4f32.nxv4f32.i64.i64(i64 3, <vscale x 4 x float> %2, <vscale x 4 x float> %1, i64 %0, i64 %3)
// CHECK:   ret <vscale x 4 x float> %5
// CHECK: }
llvm.func @ternary_xvv(%arg0: i64, %arg1: vector<[4]xf32>, %arg2: vector<[4]xf32>, %arg3: i64) -> vector<[4]xf32> {
  %0 = "vcix.intrin.ternary"(%arg0, %arg1, %arg2, %arg3) <{opcode = 3 : i64}> : (i64, vector<[4]xf32>, vector<[4]xf32>, i64) -> vector<[4]xf32>
  llvm.return %0 : vector<[4]xf32>
}

// CHECK-LABEL: define void @ternary_fvv_ro(float %0, <vscale x 4 x float> %1, <vscale x 4 x float> %2, i64 %3) {
// CHECK:   call void @llvm.riscv.sf.vc.fvv.se.i64.nxv4f32.nxv4f32.f32.i64(i64 1, <vscale x 4 x float> %2, <vscale x 4 x float> %1, float %0, i64 %3)
// CHECK:   ret void
// CHECK: }
llvm.func @ternary_fvv_ro(%arg0: f32, %arg1: vector<[4]xf32>, %arg2: vector<[4]xf32>, %arg3: i64) {
  "vcix.intrin.ternary.ro"(%arg0, %arg1, %arg2, %arg3) <{opcode = 1 : i64}> : (f32, vector<[4]xf32>, vector<[4]xf32>, i64) -> ()
  llvm.return
}

// CHECK-LABEL: define <vscale x 4 x float> @ternary_fvv(float %0, <vscale x 4 x float> %1, <vscale x 4 x float> %2, i64 %3) {
// CHECK:   %5 = call <vscale x 4 x float> @llvm.riscv.sf.vc.v.fvv.se.nxv4f32.i64.nxv4f32.nxv4f32.f32.i64(i64 1, <vscale x 4 x float> %2, <vscale x 4 x float> %1, float %0, i64 %3)
// CHECK:   ret <vscale x 4 x float> %5
// CHECK: }
llvm.func @ternary_fvv(%arg0: f32, %arg1: vector<[4]xf32>, %arg2: vector<[4]xf32>, %arg3: i64) -> vector<[4]xf32> {
  %0 = "vcix.intrin.ternary"(%arg0, %arg1, %arg2, %arg3) <{opcode = 1 : i64}> : (f32, vector<[4]xf32>, vector<[4]xf32>, i64) -> vector<[4]xf32>
  llvm.return %0 : vector<[4]xf32>
}

// CHECK-LABEL: define void @ternary_ivv_ro(<vscale x 4 x float> %0, <vscale x 4 x float> %1, i64 %2) {
// CHECK:   call void @llvm.riscv.sf.vc.xvv.se.i64.nxv4f32.nxv4f32.i64.i64(i64 3, <vscale x 4 x float> %1, <vscale x 4 x float> %0, i64 1, i64 %2)
// CHECK:   ret void
// CHECK: }
llvm.func @ternary_ivv_ro(%arg0: vector<[4]xf32>, %arg1: vector<[4]xf32>, %arg2: i64) {
  %0 = llvm.mlir.constant(1 : i5) : i5
  %1 = llvm.zext %0 : i5 to i64
  "vcix.intrin.ternary.ro"(%1, %arg0, %arg1, %arg2) <{opcode = 3 : i64}> : (i64, vector<[4]xf32>, vector<[4]xf32>, i64) -> ()
  llvm.return
}

// CHECK-LABEL: define <vscale x 4 x float> @ternary_ivv(<vscale x 4 x float> %0, <vscale x 4 x float> %1, i64 %2) {
// CHECK:   %4 = call <vscale x 4 x float> @llvm.riscv.sf.vc.v.xvv.se.nxv4f32.i64.nxv4f32.nxv4f32.i64.i64(i64 3, <vscale x 4 x float> %1, <vscale x 4 x float> %0, i64 1, i64 %2)
// CHECK:   ret <vscale x 4 x float> %4
// CHECK: }
llvm.func @ternary_ivv(%arg0: vector<[4]xf32>, %arg1: vector<[4]xf32>, %arg2: i64) -> vector<[4]xf32> {
  %0 = llvm.mlir.constant(1 : i5) : i5
  %1 = llvm.zext %0 : i5 to i64
  %2 = "vcix.intrin.ternary"(%1, %arg0, %arg1, %arg2) <{opcode = 3 : i64}> : (i64, vector<[4]xf32>, vector<[4]xf32>, i64) -> vector<[4]xf32>
  llvm.return %2 : vector<[4]xf32>
}

// CHECK-LABEL: define void @wide_ternary_vvw_ro(<vscale x 4 x float> %0, <vscale x 4 x float> %1, <vscale x 4 x double> %2, i64 %3) {
// CHECK:   call void @llvm.riscv.sf.vc.vvw.se.i64.nxv4f64.nxv4f32.nxv4f32.i64(i64 3, <vscale x 4 x double> %2, <vscale x 4 x float> %1, <vscale x 4 x float> %0, i64 %3)
// CHECK:   ret void
// CHECK: }
llvm.func @wide_ternary_vvw_ro(%arg0: vector<[4]xf32>, %arg1: vector<[4]xf32>, %arg2: vector<[4]xf64>, %arg3: i64) {
  "vcix.intrin.wide.ternary.ro"(%arg0, %arg1, %arg2, %arg3) <{opcode = 3 : i64}> : (vector<[4]xf32>, vector<[4]xf32>, vector<[4]xf64>, i64) -> ()
  llvm.return
}

// CHECK-LABEL: define <vscale x 4 x double> @wide_ternary_vvw(<vscale x 4 x float> %0, <vscale x 4 x float> %1, <vscale x 4 x double> %2, i64 %3) {
// CHECK:   %5 = call <vscale x 4 x double> @llvm.riscv.sf.vc.v.vvw.se.nxv4f64.i64.nxv4f64.nxv4f32.nxv4f32.i64(i64 3, <vscale x 4 x double> %2, <vscale x 4 x float> %1, <vscale x 4 x float> %0, i64 %3)
// CHECK:   ret <vscale x 4 x double> %5
// CHECK: }
llvm.func @wide_ternary_vvw(%arg0: vector<[4]xf32>, %arg1: vector<[4]xf32>, %arg2: vector<[4]xf64>, %arg3: i64) -> vector<[4]xf64> {
  %0 = "vcix.intrin.wide.ternary"(%arg0, %arg1, %arg2, %arg3) <{opcode = 3 : i64}> : (vector<[4]xf32>, vector<[4]xf32>, vector<[4]xf64>, i64) -> vector<[4]xf64>
  llvm.return %0 : vector<[4]xf64>
}

// CHECK-LABEL: define void @wide_ternary_xvw_ro(i64 %0, <vscale x 4 x float> %1, <vscale x 4 x double> %2, i64 %3) {
// CHECK:   call void @llvm.riscv.sf.vc.xvw.se.i64.nxv4f64.nxv4f32.i64.i64(i64 3, <vscale x 4 x double> %2, <vscale x 4 x float> %1, i64 %0, i64 %3)
// CHECK:   ret void
// CHECK: }
llvm.func @wide_ternary_xvw_ro(%arg0: i64, %arg1: vector<[4]xf32>, %arg2: vector<[4]xf64>, %arg3: i64) {
  "vcix.intrin.wide.ternary.ro"(%arg0, %arg1, %arg2, %arg3) <{opcode = 3 : i64}> : (i64, vector<[4]xf32>, vector<[4]xf64>, i64) -> ()
  llvm.return
}

// CHECK-LABEL: define <vscale x 4 x double> @wide_ternary_xvw(i64 %0, <vscale x 4 x float> %1, <vscale x 4 x double> %2, i64 %3) {
// CHECK:   %5 = call <vscale x 4 x double> @llvm.riscv.sf.vc.v.xvw.se.nxv4f64.i64.nxv4f64.nxv4f32.i64.i64(i64 3, <vscale x 4 x double> %2, <vscale x 4 x float> %1, i64 %0, i64 %3)
// CHECK:   ret <vscale x 4 x double> %5
// CHECK: }
llvm.func @wide_ternary_xvw(%arg0: i64, %arg1: vector<[4]xf32>, %arg2: vector<[4]xf64>, %arg3: i64) -> vector<[4]xf64> {
  %0 = "vcix.intrin.wide.ternary"(%arg0, %arg1, %arg2, %arg3) <{opcode = 3 : i64}> : (i64, vector<[4]xf32>, vector<[4]xf64>, i64) -> vector<[4]xf64>
  llvm.return %0 : vector<[4]xf64>
}

// CHECK-LABEL: define void @wide_ternary_fvw_ro(float %0, <vscale x 4 x float> %1, <vscale x 4 x double> %2, i64 %3) {
// CHECK:   call void @llvm.riscv.sf.vc.fvw.se.i64.nxv4f64.nxv4f32.f32.i64(i64 1, <vscale x 4 x double> %2, <vscale x 4 x float> %1, float %0, i64 %3)
// CHECK:   ret void
// CHECK: }
llvm.func @wide_ternary_fvw_ro(%arg0: f32, %arg1: vector<[4]xf32>, %arg2: vector<[4]xf64>, %arg3: i64) {
  "vcix.intrin.wide.ternary.ro"(%arg0, %arg1, %arg2, %arg3) <{opcode = 1 : i64}> : (f32, vector<[4]xf32>, vector<[4]xf64>, i64) -> ()
  llvm.return
}

// CHECK-LABEL: define <vscale x 4 x double> @wide_ternary_fvw(float %0, <vscale x 4 x float> %1, <vscale x 4 x double> %2, i64 %3) {
// CHECK:   %5 = call <vscale x 4 x double> @llvm.riscv.sf.vc.v.fvw.se.nxv4f64.i64.nxv4f64.nxv4f32.f32.i64(i64 1, <vscale x 4 x double> %2, <vscale x 4 x float> %1, float %0, i64 %3)
// CHECK:   ret <vscale x 4 x double> %2
// CHECK: }
llvm.func @wide_ternary_fvw(%arg0: f32, %arg1: vector<[4]xf32>, %arg2: vector<[4]xf64>, %arg3: i64) -> vector<[4]xf64> {
  %0 = "vcix.intrin.wide.ternary"(%arg0, %arg1, %arg2, %arg3) <{opcode = 1 : i64}> : (f32, vector<[4]xf32>, vector<[4]xf64>, i64) -> vector<[4]xf64>
  llvm.return %arg2 : vector<[4]xf64>
}

// CHECK-LABEL: define void @wide_ternary_ivw_ro(<vscale x 4 x float> %0, <vscale x 4 x double> %1, i64 %2) {
// CHECK:   call void @llvm.riscv.sf.vc.xvw.se.i64.nxv4f64.nxv4f32.i64.i64(i64 3, <vscale x 4 x double> %1, <vscale x 4 x float> %0, i64 1, i64 %2)
// CHECK:   ret void
// CHECK: }
llvm.func @wide_ternary_ivw_ro(%arg0: vector<[4]xf32>, %arg1: vector<[4]xf64>, %arg2: i64) {
  %0 = llvm.mlir.constant(1 : i5) : i5
  %1 = llvm.zext %0 : i5 to i64
  "vcix.intrin.wide.ternary.ro"(%1, %arg0, %arg1, %arg2) <{opcode = 3 : i64}> : (i64, vector<[4]xf32>, vector<[4]xf64>, i64) -> ()
  llvm.return
}

// CHECK-LABEL: define <vscale x 4 x double> @wide_ternary_ivv(<vscale x 4 x float> %0, <vscale x 4 x double> %1, i64 %2) {
// CHECK:   %4 = call <vscale x 4 x double> @llvm.riscv.sf.vc.v.xvw.se.nxv4f64.i64.nxv4f64.nxv4f32.i64.i64(i64 3, <vscale x 4 x double> %1, <vscale x 4 x float> %0, i64 1, i64 %2)
// CHECK:   ret <vscale x 4 x double> %1
// CHECK: }
llvm.func @wide_ternary_ivv(%arg0: vector<[4]xf32>, %arg1: vector<[4]xf64>, %arg2: i64) -> vector<[4]xf64> {
  %0 = llvm.mlir.constant(1 : i5) : i5
  %1 = llvm.zext %0 : i5 to i64
  %2 = "vcix.intrin.wide.ternary"(%1, %arg0, %arg1, %arg2) <{opcode = 3 : i64}> : (i64, vector<[4]xf32>, vector<[4]xf64>, i64) -> vector<[4]xf64>
  llvm.return %arg1 : vector<[4]xf64>
}

// CHECK-LABEL: define void @fixed_binary_vv_ro(<4 x float> %0, <4 x float> %1) {
// CHECK:   call void @llvm.riscv.sf.vc.vv.se.i64.v4f32.v4f32.i64(i64 3, i64 30, <4 x float> %1, <4 x float> %0, i64 4)
// CHECK:   ret void
// CHECK: }
llvm.func @fixed_binary_vv_ro(%arg0: vector<4xf32>, %arg1: vector<4xf32>) {
  "vcix.intrin.binary.ro"(%arg0, %arg1) <{opcode = 3 : i64, rd = 30 : i64}> : (vector<4xf32>, vector<4xf32>) -> ()
  llvm.return
}

// CHECK-LABEL: define <4 x float> @fixed_binary_vv(<4 x float> %0, <4 x float> %1) {
// CHECK:   %3 = call <4 x float> @llvm.riscv.sf.vc.v.vv.se.v4f32.i64.v4f32.v4f32.i64(i64 3, <4 x float> %1, <4 x float> %0, i64 4)
// CHECK:   ret <4 x float> %3
// CHECK: }
llvm.func @fixed_binary_vv(%arg0: vector<4xf32>, %arg1: vector<4xf32>) -> vector<4xf32> {
  %0 = "vcix.intrin.binary"(%arg0, %arg1) <{opcode = 3 : i64}> : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
  llvm.return %0 : vector<4xf32>
}

// CHECK-LABEL: define void @fixed_binary_xv_ro(i64 %0, <4 x float> %1) {
// CHECK:   call void @llvm.riscv.sf.vc.xv.se.i64.v4f32.i64.i64(i64 3, i64 30, <4 x float> %1, i64 %0, i64 4)
// CHECK:   ret void
// CHECK: }
llvm.func @fixed_binary_xv_ro(%arg0: i64, %arg1: vector<4xf32>) {
  "vcix.intrin.binary.ro"(%arg0, %arg1) <{opcode = 3 : i64, rd = 30 : i64}> : (i64, vector<4xf32>) -> ()
  llvm.return
}

// CHECK-LABEL: define <4 x float> @fixed_binary_xv(i64 %0, <4 x float> %1) {
// CHECK:   %3 = call <4 x float> @llvm.riscv.sf.vc.v.xv.se.v4f32.i64.v4f32.i64.i64(i64 3, <4 x float> %1, i64 %0, i64 4)
// CHECK:   ret <4 x float> %3
// CHECK: }
llvm.func @fixed_binary_xv(%arg0: i64, %arg1: vector<4xf32>) -> vector<4xf32> {
  %0 = "vcix.intrin.binary"(%arg0, %arg1) <{opcode = 3 : i64}> : (i64, vector<4xf32>) -> vector<4xf32>
  llvm.return %0 : vector<4xf32>
}

// CHECK-LABEL: define void @fixed_binary_fv_ro(float %0, <4 x float> %1) {
// CHECK:   call void @llvm.riscv.sf.vc.fv.se.i64.v4f32.f32.i64(i64 1, i64 30, <4 x float> %1, float %0, i64 4)
// CHECK:   ret void
// CHECK: }
llvm.func @fixed_binary_fv_ro(%arg0: f32, %arg1: vector<4xf32>) {
  "vcix.intrin.binary.ro"(%arg0, %arg1) <{opcode = 1 : i64, rd = 30 : i64}> : (f32, vector<4xf32>) -> ()
  llvm.return
}

// CHECK-LABEL: define <4 x float> @fixed_binary_fv(float %0, <4 x float> %1) {
// CHECK:   %3 = call <4 x float> @llvm.riscv.sf.vc.v.fv.se.v4f32.i64.v4f32.f32.i64(i64 1, <4 x float> %1, float %0, i64 4)
// CHECK:   ret <4 x float> %3
// CHECK: }
llvm.func @fixed_binary_fv(%arg0: f32, %arg1: vector<4xf32>) -> vector<4xf32> {
  %0 = "vcix.intrin.binary"(%arg0, %arg1) <{opcode = 1 : i64}> : (f32, vector<4xf32>) -> vector<4xf32>
  llvm.return %0 : vector<4xf32>
}

// CHECK-LABEL: define void @fixed_binary_iv_ro(<4 x float> %0) {
// CHECK:   call void @llvm.riscv.sf.vc.xv.se.i64.v4f32.i64.i64(i64 3, i64 30, <4 x float> %0, i64 1, i64 4)
// CHECK:   ret void
// CHECK: }
llvm.func @fixed_binary_iv_ro(%arg0: vector<4xf32>) {
  %0 = llvm.mlir.constant(1 : i5) : i5
  %1 = llvm.zext %0 : i5 to i64
  "vcix.intrin.binary.ro"(%1, %arg0) <{opcode = 3 : i64, rd = 30 : i64}> : (i64, vector<4xf32>) -> ()
  llvm.return
}

// CHECK-LABEL: define <4 x float> @fixed_binary_iv(<4 x float> %0) {
// CHECK:   %2 = call <4 x float> @llvm.riscv.sf.vc.v.xv.se.v4f32.i64.v4f32.i64.i64(i64 3, <4 x float> %0, i64 1, i64 4)
// CHECK:   ret <4 x float> %2
// CHECK: }
llvm.func @fixed_binary_iv(%arg0: vector<4xf32>) -> vector<4xf32> {
  %0 = llvm.mlir.constant(1 : i5) : i5
  %1 = llvm.zext %0 : i5 to i64
  %2 = "vcix.intrin.binary"(%1, %arg0) <{opcode = 3 : i64}> : (i64, vector<4xf32>) -> vector<4xf32>
  llvm.return %2 : vector<4xf32>
}

// CHECK-LABEL: define void @fixed_ternary_vvv_ro(<4 x float> %0, <4 x float> %1, <4 x float> %2) {
// CHECK:   call void @llvm.riscv.sf.vc.vvv.se.i64.v4f32.v4f32.v4f32.i64(i64 3, <4 x float> %2, <4 x float> %1, <4 x float> %0, i64 4)
// CHECK:   ret void
// CHECK: }
llvm.func @fixed_ternary_vvv_ro(%arg0: vector<4xf32>, %arg1: vector<4xf32>, %arg2: vector<4xf32>) {
  "vcix.intrin.ternary.ro"(%arg0, %arg1, %arg2) <{opcode = 3 : i64}> : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> ()
  llvm.return
}

// CHECK-LABEL: define <4 x float> @fixed_ternary_vvv(<4 x float> %0, <4 x float> %1, <4 x float> %2) {
// CHECK:   %4 = call <4 x float> @llvm.riscv.sf.vc.v.vvv.se.v4f32.i64.v4f32.v4f32.v4f32.i64(i64 3, <4 x float> %2, <4 x float> %1, <4 x float> %0, i64 4)
// CHECK:   ret <4 x float> %4
// CHECK: }
llvm.func @fixed_ternary_vvv(%arg0: vector<4xf32>, %arg1: vector<4xf32>, %arg2: vector<4xf32>) -> vector<4xf32> {
  %0 = "vcix.intrin.ternary"(%arg0, %arg1, %arg2) <{opcode = 3 : i64}> : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
  llvm.return %0 : vector<4xf32>
}

// CHECK-LABEL: define void @fixed_ternary_xvv_ro(i64 %0, <4 x float> %1, <4 x float> %2) {
// CHECK:   call void @llvm.riscv.sf.vc.xvv.se.i64.v4f32.v4f32.i64.i64(i64 3, <4 x float> %2, <4 x float> %1, i64 %0, i64 4)
// CHECK:   ret void
// CHECK: }
llvm.func @fixed_ternary_xvv_ro(%arg0: i64, %arg1: vector<4xf32>, %arg2: vector<4xf32>) {
  "vcix.intrin.ternary.ro"(%arg0, %arg1, %arg2) <{opcode = 3 : i64}> : (i64, vector<4xf32>, vector<4xf32>) -> ()
  llvm.return
}

// CHECK-LABEL: define <4 x float> @fixed_ternary_xvv(i64 %0, <4 x float> %1, <4 x float> %2) {
// CHECK:   %4 = call <4 x float> @llvm.riscv.sf.vc.v.xvv.se.v4f32.i64.v4f32.v4f32.i64.i64(i64 3, <4 x float> %2, <4 x float> %1, i64 %0, i64 4)
// CHECK:   ret <4 x float> %4
// CHECK: }
llvm.func @fixed_ternary_xvv(%arg0: i64, %arg1: vector<4xf32>, %arg2: vector<4xf32>) -> vector<4xf32> {
  %0 = "vcix.intrin.ternary"(%arg0, %arg1, %arg2) <{opcode = 3 : i64}> : (i64, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
  llvm.return %0 : vector<4xf32>
}

// CHECK-LABEL: define void @fixed_ternary_fvv_ro(float %0, <4 x float> %1, <4 x float> %2) {
// CHECK:   call void @llvm.riscv.sf.vc.fvv.se.i64.v4f32.v4f32.f32.i64(i64 1, <4 x float> %2, <4 x float> %1, float %0, i64 4)
// CHECK:   ret void
// CHECK: }
llvm.func @fixed_ternary_fvv_ro(%arg0: f32, %arg1: vector<4xf32>, %arg2: vector<4xf32>) {
  "vcix.intrin.ternary.ro"(%arg0, %arg1, %arg2) <{opcode = 1 : i64}> : (f32, vector<4xf32>, vector<4xf32>) -> ()
  llvm.return
}

// CHECK-LABEL: define <4 x float> @fixed_ternary_fvv(float %0, <4 x float> %1, <4 x float> %2) {
// CHECK:   %4 = call <4 x float> @llvm.riscv.sf.vc.v.fvv.se.v4f32.i64.v4f32.v4f32.f32.i64(i64 1, <4 x float> %2, <4 x float> %1, float %0, i64 4)
// CHECK:   ret <4 x float> %4
// CHECK: }
llvm.func @fixed_ternary_fvv(%arg0: f32, %arg1: vector<4xf32>, %arg2: vector<4xf32>) -> vector<4xf32> {
  %0 = "vcix.intrin.ternary"(%arg0, %arg1, %arg2) <{opcode = 1 : i64}> : (f32, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
  llvm.return %0 : vector<4xf32>
}

// CHECK-LABEL: define void @fixed_ternary_ivv_ro(<4 x float> %0, <4 x float> %1) {
// CHECK:   call void @llvm.riscv.sf.vc.xvv.se.i64.v4f32.v4f32.i64.i64(i64 3, <4 x float> %1, <4 x float> %0, i64 1, i64 4)
// CHECK:   ret void
// CHECK: }
llvm.func @fixed_ternary_ivv_ro(%arg0: vector<4xf32>, %arg1: vector<4xf32>) {
  %0 = llvm.mlir.constant(1 : i5) : i5
  %1 = llvm.zext %0 : i5 to i64
  "vcix.intrin.ternary.ro"(%1, %arg0, %arg1) <{opcode = 3 : i64}> : (i64, vector<4xf32>, vector<4xf32>) -> ()
  llvm.return
}

// CHECK-LABEL: define <4 x float> @fixed_ternary_ivv(<4 x float> %0, <4 x float> %1) {
// CHECK:   %3 = call <4 x float> @llvm.riscv.sf.vc.v.xvv.se.v4f32.i64.v4f32.v4f32.i64.i64(i64 3, <4 x float> %1, <4 x float> %0, i64 1, i64 4)
// CHECK:   ret <4 x float> %3
// CHECK: }
llvm.func @fixed_ternary_ivv(%arg0: vector<4xf32>, %arg1: vector<4xf32>) -> vector<4xf32> {
  %0 = llvm.mlir.constant(1 : i5) : i5
  %1 = llvm.zext %0 : i5 to i64
  %2 = "vcix.intrin.ternary"(%1, %arg0, %arg1) <{opcode = 3 : i64}> : (i64, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
  llvm.return %2 : vector<4xf32>
}

// CHECK-LABEL: define void @fixed_wide_ternary_vvw_ro(<4 x float> %0, <4 x float> %1, <4 x double> %2) {
// CHECK:   call void @llvm.riscv.sf.vc.vvw.se.i64.v4f64.v4f32.v4f32.i64(i64 3, <4 x double> %2, <4 x float> %1, <4 x float> %0, i64 4)
// CHECK:   ret void
// CHECK: }
llvm.func @fixed_wide_ternary_vvw_ro(%arg0: vector<4xf32>, %arg1: vector<4xf32>, %arg2: vector<4xf64>) {
  "vcix.intrin.wide.ternary.ro"(%arg0, %arg1, %arg2) <{opcode = 3 : i64}> : (vector<4xf32>, vector<4xf32>, vector<4xf64>) -> ()
  llvm.return
}

// CHECK-LABEL: define <4 x double> @fixed_wide_ternary_vvw(<4 x float> %0, <4 x float> %1, <4 x double> %2) {
// CHECK:   %4 = call <4 x double> @llvm.riscv.sf.vc.v.vvw.se.v4f64.i64.v4f64.v4f32.v4f32.i64(i64 3, <4 x double> %2, <4 x float> %1, <4 x float> %0, i64 4)
// CHECK:   ret <4 x double> %4
// CHECK: }
llvm.func @fixed_wide_ternary_vvw(%arg0: vector<4xf32>, %arg1: vector<4xf32>, %arg2: vector<4xf64>) -> vector<4xf64> {
  %0 = "vcix.intrin.wide.ternary"(%arg0, %arg1, %arg2) <{opcode = 3 : i64}> : (vector<4xf32>, vector<4xf32>, vector<4xf64>) -> vector<4xf64>
  llvm.return %0 : vector<4xf64>
}

// CHECK-LABEL: define void @fixed_wide_ternary_xvw_ro(i64 %0, <4 x float> %1, <4 x double> %2) {
// CHECK:   call void @llvm.riscv.sf.vc.xvw.se.i64.v4f64.v4f32.i64.i64(i64 3, <4 x double> %2, <4 x float> %1, i64 %0, i64 4)
// CHECK:   ret void
// CHECK: }
llvm.func @fixed_wide_ternary_xvw_ro(%arg0: i64, %arg1: vector<4xf32>, %arg2: vector<4xf64>) {
  "vcix.intrin.wide.ternary.ro"(%arg0, %arg1, %arg2) <{opcode = 3 : i64}> : (i64, vector<4xf32>, vector<4xf64>) -> ()
  llvm.return
}

// CHECK-LABEL: define <4 x double> @fixed_wide_ternary_xvw(i64 %0, <4 x float> %1, <4 x double> %2) {
// CHECK:   %4 = call <4 x double> @llvm.riscv.sf.vc.v.xvw.se.v4f64.i64.v4f64.v4f32.i64.i64(i64 3, <4 x double> %2, <4 x float> %1, i64 %0, i64 4)
// CHECK:   ret <4 x double> %4
// CHECK: }
llvm.func @fixed_wide_ternary_xvw(%arg0: i64, %arg1: vector<4xf32>, %arg2: vector<4xf64>) -> vector<4xf64> {
  %0 = "vcix.intrin.wide.ternary"(%arg0, %arg1, %arg2) <{opcode = 3 : i64}> : (i64, vector<4xf32>, vector<4xf64>) -> vector<4xf64>
  llvm.return %0 : vector<4xf64>
}

// CHECK-LABEL: define void @fixed_wide_ternary_fvw_ro(float %0, <4 x float> %1, <4 x double> %2) {
// CHECK:   call void @llvm.riscv.sf.vc.fvw.se.i64.v4f64.v4f32.f32.i64(i64 1, <4 x double> %2, <4 x float> %1, float %0, i64 4)
// CHECK:   ret void
// CHECK: }
llvm.func @fixed_wide_ternary_fvw_ro(%arg0: f32, %arg1: vector<4xf32>, %arg2: vector<4xf64>) {
  "vcix.intrin.wide.ternary.ro"(%arg0, %arg1, %arg2) <{opcode = 1 : i64}> : (f32, vector<4xf32>, vector<4xf64>) -> ()
  llvm.return
}

// CHECK-LABEL: define <4 x double> @fixed_wide_ternary_fvw(float %0, <4 x float> %1, <4 x double> %2) {
// CHECK:   %4 = call <4 x double> @llvm.riscv.sf.vc.v.fvw.se.v4f64.i64.v4f64.v4f32.f32.i64(i64 1, <4 x double> %2, <4 x float> %1, float %0, i64 4)
// CHECK:   ret <4 x double> %2
// CHECK: }
llvm.func @fixed_wide_ternary_fvw(%arg0: f32, %arg1: vector<4xf32>, %arg2: vector<4xf64>) -> vector<4xf64> {
  %0 = "vcix.intrin.wide.ternary"(%arg0, %arg1, %arg2) <{opcode = 1 : i64}> : (f32, vector<4xf32>, vector<4xf64>) -> vector<4xf64>
  llvm.return %arg2 : vector<4xf64>
}

// CHECK-LABEL: define void @fixed_wide_ternary_ivw_ro(<4 x float> %0, <4 x double> %1) {
// CHECK:   call void @llvm.riscv.sf.vc.xvw.se.i64.v4f64.v4f32.i64.i64(i64 3, <4 x double> %1, <4 x float> %0, i64 1, i64 4)
// CHECK:   ret void
// CHECK: }
llvm.func @fixed_wide_ternary_ivw_ro(%arg0: vector<4xf32>, %arg1: vector<4xf64>) {
  %0 = llvm.mlir.constant(1 : i5) : i5
  %1 = llvm.zext %0 : i5 to i64
  "vcix.intrin.wide.ternary.ro"(%1, %arg0, %arg1) <{opcode = 3 : i64}> : (i64, vector<4xf32>, vector<4xf64>) -> ()
  llvm.return
}

// CHECK-LABEL: define <4 x double> @fixed_wide_ternary_ivv(<4 x float> %0, <4 x double> %1) {
// CHECK:   %3 = call <4 x double> @llvm.riscv.sf.vc.v.xvw.se.v4f64.i64.v4f64.v4f32.i64.i64(i64 3, <4 x double> %1, <4 x float> %0, i64 1, i64 4)
// CHECK:   ret <4 x double> %1
// CHECK: }
llvm.func @fixed_wide_ternary_ivv(%arg0: vector<4xf32>, %arg1: vector<4xf64>) -> vector<4xf64> {
  %0 = llvm.mlir.constant(1 : i5) : i5
  %1 = llvm.zext %0 : i5 to i64
  %2 = "vcix.intrin.wide.ternary"(%1, %arg0, %arg1) <{opcode = 3 : i64}> : (i64, vector<4xf32>, vector<4xf64>) -> vector<4xf64>
  llvm.return %arg1 : vector<4xf64>
}
