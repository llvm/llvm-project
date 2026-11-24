// RUN: mlir-opt --convert-xevm-to-llvm --split-input-file %s | FileCheck %s

// Driver for extending HandleVectorExtractPattern to hoist contiguous-slice
// shuffles through n-ary elementwise ops (here the binary `llvm.fdiv`).
//
// This is the exact code sequence taken from the WG-level mxfp GEMM
// (simple_mxfp_gemm_quantizeA_F4.mlir) at the point it is fed to
// convert-xevm-to-llvm: the f8E8M0 scale is expanded into wide integer/float
// arithmetic (zext/shl/icmp/select/bitcast/fptrunc/fpext/fdiv/fptrunc on
// vector<32x...>), then sliced into two contiguous vector<16xbf16> halves that
// feed `xevm.truncf` (bf16 -> e2m1), which are assembled into the `a` operand
// of `xevm.mma_mx`.
//
// The `[0..15]`/`[16..31]` slices already sit below the wide chain; the
// existing pattern hoists them through the unary `fptrunc` but gets stuck at
// the binary `fdiv`, so the wide vector<32x...> compute survives and the
// SPIR-V backend later fails to legalize the wide `fpext`/`fdiv`.
//
// Once the pattern hoists through n-ary elementwise ops, the whole scale chain
// must narrow to the 16-wide slice width and no vector<32x...> compute op may
// remain.

// CHECK-LABEL: llvm.func @mxfp_scale_binary_hoist
// The wide f8E8M0 scale chain must be narrowed to the 16-wide slice: no wide
// (vector<32x...>) elementwise compute op may survive. Wide function arguments,
// constants, the packed `b` payload and data-movement shuffles are unaffected.
// CHECK-NOT:     llvm.fdiv {{.*}} : vector<32xf32>
// CHECK-NOT:     llvm.fpext {{.*}} : vector<32xbf16>
// CHECK-NOT:     llvm.fptrunc {{.*}} : vector<32xf32>
// CHECK-NOT:     llvm.zext {{.*}} : vector<32xi8> to vector<32xi32>
// CHECK-NOT:     llvm.shl {{.*}} : vector<32xi32>
// CHECK-NOT:     llvm.select {{.*}} : vector<32xi1>, vector<32xi32>
// CHECK-NOT:     llvm.icmp {{.*}} : vector<32xi8>
// CHECK:         llvm.fdiv {{.*}} : vector<16xf32>
llvm.func @mxfp_scale_binary_hoist(
    %scale_src: vector<32xi8>,
    %num: vector<32xf32>,
    %b: vector<32xi8>,
    %scale_a: vector<2xi8>,
    %scale_b: vector<2xi8>,
    %c: vector<8xf32>) -> vector<8xf32> {
  %shl_amt   = llvm.mlir.constant(dense<23> : vector<32xi32>) : vector<32xi32>
  %ones_i8   = llvm.mlir.constant(dense<-1> : vector<32xi8>)  : vector<32xi8>
  %ones_i32  = llvm.mlir.constant(dense<-1> : vector<32xi32>) : vector<32xi32>

  // f8E8M0 scale expansion into wide arithmetic.
  %e         = llvm.zext %scale_src : vector<32xi8> to vector<32xi32>
  %sh        = llvm.shl %e, %shl_amt : vector<32xi32>
  %cmp       = llvm.icmp "eq" %scale_src, %ones_i8 : vector<32xi8>
  %sel       = llvm.select %cmp, %ones_i32, %sh : vector<32xi1>, vector<32xi32>
  %scale_f32 = llvm.bitcast %sel : vector<32xi32> to vector<32xf32>
  %scale_bf  = llvm.fptrunc %scale_f32 : vector<32xf32> to vector<32xbf16>
  %scale_ext = llvm.fpext %scale_bf fastmath<contract> : vector<32xbf16> to vector<32xf32>
  %div       = llvm.fdiv %num, %scale_ext : vector<32xf32>
  %div_bf    = llvm.fptrunc %div fastmath<contract> : vector<32xf32> to vector<32xbf16>

  // Two contiguous 16-wide slices feeding xevm.truncf (bf16 -> e2m1/fp4).
  %lo  = llvm.shufflevector %div_bf, %div_bf [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<32xbf16>
  %lo4 = xevm.truncf %lo {src_etype = bf16, dst_etype = e2m1} : (vector<16xbf16>) -> vector<8xi8>
  %hi  = llvm.shufflevector %div_bf, %div_bf [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<32xbf16>
  %hi4 = xevm.truncf %hi {src_etype = bf16, dst_etype = e2m1} : (vector<16xbf16>) -> vector<8xi8>

  // Assemble the packed fp4 `a` operand (16xi8) and feed the scaled MMA.
  %a = llvm.shufflevector %lo4, %hi4 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8xi8>
  %r = xevm.mma_mx %a, %b, %scale_a, %scale_b, %c
        {shape = <m = 8, n = 16, k = 64>, types = <d = f32, a = e2m1, b = e2m1, c = f32>}
        : (vector<16xi8>, vector<32xi8>, vector<2xi8>, vector<2xi8>, vector<8xf32>) -> vector<8xf32>
  llvm.return %r : vector<8xf32>
}

// -----

// Negative: a contiguous slice of a packed sub-byte `xevm.truncf` result must
// NOT be hoisted into the truncf. The truncf changes the element count
// (16 bf16 -> 8 packed i8) and is not a same-length elementwise op, so the
// n-ary guard rejects it: the bf16 input is never fragmented into narrower
// bf16 compute (the slice stays in the packed representation instead).
// CHECK-LABEL: llvm.func @packed_subbyte_not_split
// CHECK-NOT:     vector<4xbf16>
// CHECK-NOT:     vector<8xbf16>
// CHECK:         llvm.return
llvm.func @packed_subbyte_not_split(%src: vector<16xbf16>) -> vector<4xi8> {
  %p = xevm.truncf %src {src_etype = bf16, dst_etype = e2m1} : (vector<16xbf16>) -> vector<8xi8>
  %s = llvm.shufflevector %p, %p [0, 1, 2, 3] : vector<8xi8>
  llvm.return %s : vector<4xi8>
}
