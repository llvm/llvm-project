// RUN: mlir-opt --split-input-file -convert-xegpu-to-xevm %s | FileCheck %s

// f8E8M0FNU <-> bf16/f16 extf/truncf have no direct lowering, so they are
// expanded into integer bit manipulation. f8E8M0FNU is mapped to i8 by the
// type converter.

// CHECK-LABEL: gpu.func @extf_e8m0_to_bf16
gpu.module @extf_bf16 [#xevm.target<chip = "cri">] {
  gpu.func @extf_e8m0_to_bf16(%s: vector<2xf8E8M0FNU>, %m: memref<2xbf16>) kernel {
    // bf16 shares E8M0's 8-bit exponent (bias 127): shift the exponent into
    // place, with a NaN fixup (0xFF would otherwise become +inf; 0x7FC0=32704).
    // CHECK: vector.bitcast %{{.*}} : vector<2xf8E8M0FNU> to vector<2xi8>
    // CHECK: arith.extui %{{.*}} : vector<2xi8> to vector<2xi16>
    // CHECK: arith.constant dense<7> : vector<2xi16>
    // CHECK: arith.shli %{{.*}}, %{{.*}} : vector<2xi16>
    // CHECK: arith.constant dense<32704> : vector<2xi16>
    // CHECK: arith.cmpi eq, %{{.*}}, %{{.*}} : vector<2xi16>
    // CHECK: arith.select %{{.*}}, %{{.*}}, %{{.*}} : vector<2xi1>, vector<2xi16>
    // CHECK: arith.bitcast %{{.*}} : vector<2xi16> to vector<2xbf16>
    %r = arith.extf %s : vector<2xf8E8M0FNU> to vector<2xbf16>
    %c0 = arith.constant 0 : index
    vector.store %r, %m[%c0] : memref<2xbf16>, vector<2xbf16>
    gpu.return
  }
}

// -----

// CHECK-LABEL: gpu.func @extf_e8m0_to_f16
gpu.module @extf_f16 [#xevm.target<chip = "cri">] {
  gpu.func @extf_e8m0_to_f16(%s: vector<2xf8E8M0FNU>, %m: memref<2xf16>) kernel {
    // f16 has a 5-bit exponent (bias 15): rebias (e - 112), then saturate the
    // values f16 cannot represent: underflow (e < 113) -> 0, overflow
    // (e > 142) -> inf (0x7C00=31744), NaN (0xFF) -> NaN (0x7E00=32256).
    // CHECK: vector.bitcast %{{.*}} : vector<2xf8E8M0FNU> to vector<2xi8>
    // CHECK: arith.extui %{{.*}} : vector<2xi8> to vector<2xi16>
    // CHECK: arith.constant dense<112> : vector<2xi16>
    // CHECK: arith.subi %{{.*}}, %{{.*}} : vector<2xi16>
    // CHECK: arith.shli %{{.*}}, %{{.*}} : vector<2xi16>
    // CHECK: arith.constant dense<31744> : vector<2xi16>
    // CHECK: arith.constant dense<32256> : vector<2xi16>
    // CHECK: arith.cmpi ult, %{{.*}}, %{{.*}} : vector<2xi16>
    // CHECK: arith.select %{{.*}}, %{{.*}}, %{{.*}} : vector<2xi1>, vector<2xi16>
    // CHECK: arith.cmpi ugt, %{{.*}}, %{{.*}} : vector<2xi16>
    // CHECK: arith.select %{{.*}}, %{{.*}}, %{{.*}} : vector<2xi1>, vector<2xi16>
    // CHECK: arith.cmpi eq, %{{.*}}, %{{.*}} : vector<2xi16>
    // CHECK: arith.select %{{.*}}, %{{.*}}, %{{.*}} : vector<2xi1>, vector<2xi16>
    // CHECK: arith.bitcast %{{.*}} : vector<2xi16> to vector<2xf16>
    %r = arith.extf %s : vector<2xf8E8M0FNU> to vector<2xf16>
    %c0 = arith.constant 0 : index
    vector.store %r, %m[%c0] : memref<2xf16>, vector<2xf16>
    gpu.return
  }
}

// -----

// CHECK-LABEL: gpu.func @truncf_bf16_to_e8m0
gpu.module @truncf_bf16 [#xevm.target<chip = "cri">] {
  gpu.func @truncf_bf16_to_e8m0(%s: vector<2xbf16>, %m: memref<2xf8E8M0FNU>) kernel {
    // bf16 shares E8M0's exponent encoding: round-to-nearest-even across the 7
    // mantissa bits (bias 0x3F=63 + exponent lsb), then keep the exponent byte.
    // CHECK: arith.bitcast %{{.*}} : vector<2xbf16> to vector<2xi16>
    // CHECK: arith.constant dense<32767> : vector<2xi16>
    // CHECK: arith.andi %{{.*}}, %{{.*}} : vector<2xi16>
    // CHECK: arith.shrui %{{.*}}, %{{.*}} : vector<2xi16>
    // CHECK: arith.constant dense<63> : vector<2xi16>
    // CHECK: arith.addi %{{.*}}, %{{.*}} : vector<2xi16>
    // CHECK: arith.addi %{{.*}}, %{{.*}} : vector<2xi16>
    // CHECK: arith.shrui %{{.*}}, %{{.*}} : vector<2xi16>
    // CHECK: arith.constant dense<255> : vector<2xi16>
    // CHECK: arith.andi %{{.*}}, %{{.*}} : vector<2xi16>
    // CHECK: arith.trunci %{{.*}} : vector<2xi16> to vector<2xi8>
    // CHECK: vector.bitcast %{{.*}} : vector<2xi8> to vector<2xf8E8M0FNU>
    %r = arith.truncf %s : vector<2xbf16> to vector<2xf8E8M0FNU>
    %c0 = arith.constant 0 : index
    vector.store %r, %m[%c0] : memref<2xf8E8M0FNU>, vector<2xf8E8M0FNU>
    gpu.return
  }
}

// -----

// CHECK-LABEL: gpu.func @truncf_f16_to_e8m0
gpu.module @truncf_f16 [#xevm.target<chip = "cri">] {
  gpu.func @truncf_f16_to_e8m0(%s: vector<2xf16>, %m: memref<2xf8E8M0FNU>) kernel {
    // f16: round-to-nearest-even across the 10 mantissa bits (bias 0x1FF=511 +
    // exponent lsb), rebias the 5-bit exponent to E8M0 (e = f16_exp + 112), and
    // map f16 inf/NaN (exponent all ones == 31) to E8M0 NaN (0xFF == 255).
    // CHECK: arith.bitcast %{{.*}} : vector<2xf16> to vector<2xi16>
    // CHECK: arith.constant dense<32767> : vector<2xi16>
    // CHECK: arith.andi %{{.*}}, %{{.*}} : vector<2xi16>
    // CHECK: arith.shrui %{{.*}}, %{{.*}} : vector<2xi16>
    // CHECK: arith.constant dense<31> : vector<2xi16>
    // CHECK: arith.andi %{{.*}}, %{{.*}} : vector<2xi16>
    // CHECK: arith.andi %{{.*}}, %{{.*}} : vector<2xi16>
    // CHECK: arith.constant dense<511> : vector<2xi16>
    // CHECK: arith.addi %{{.*}}, %{{.*}} : vector<2xi16>
    // CHECK: arith.addi %{{.*}}, %{{.*}} : vector<2xi16>
    // CHECK: arith.shrui %{{.*}}, %{{.*}} : vector<2xi16>
    // CHECK: arith.constant dense<112> : vector<2xi16>
    // CHECK: arith.addi %{{.*}}, %{{.*}} : vector<2xi16>
    // CHECK: arith.constant dense<255> : vector<2xi16>
    // CHECK: arith.cmpi eq, %{{.*}}, %{{.*}} : vector<2xi16>
    // CHECK: arith.select %{{.*}}, %{{.*}}, %{{.*}} : vector<2xi1>, vector<2xi16>
    // CHECK: arith.trunci %{{.*}} : vector<2xi16> to vector<2xi8>
    // CHECK: vector.bitcast %{{.*}} : vector<2xi8> to vector<2xf8E8M0FNU>
    %r = arith.truncf %s : vector<2xf16> to vector<2xf8E8M0FNU>
    %c0 = arith.constant 0 : index
    vector.store %r, %m[%c0] : memref<2xf8E8M0FNU>, vector<2xf8E8M0FNU>
    gpu.return
  }
}
