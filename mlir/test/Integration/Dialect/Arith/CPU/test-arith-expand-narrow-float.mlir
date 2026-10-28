// Value tests for the narrow-float extf expansion. Expected values come from an
// independent model of APFloat::convert; inputs are runtime arguments, so
// nothing is constant folded.

// RUN: mlir-opt %s --arith-expand="include-f6e2m3fn=true include-f6e3m2fn=true \
// RUN:     include-f8e3m4=true include-f8e4m3=true include-f8e4m3b11fnuz=true \
// RUN:     include-f8e4m3fnuz=true include-f8e5m2fnuz=true" \
// RUN:     --convert-vector-to-scf --convert-scf-to-cf --convert-cf-to-llvm \
// RUN:     --convert-vector-to-llvm --convert-func-to-llvm \
// RUN:     --convert-arith-to-llvm --reconcile-unrealized-casts | \
// RUN:   mlir-runner -e entry --entry-point-result=void \
// RUN:               --shared-libs=%mlir_c_runner_utils | \
// RUN:   FileCheck %s --match-full-lines

// Prints the bits of a vector as unsigned integers.
func.func @print_f32(%v : vector<8xf32>) {
  %bits = arith.bitcast %v : vector<8xf32> to vector<8xi32>
  %u = vector.bitcast %bits : vector<8xi32> to vector<8xui32>
  vector.print %u : vector<8xui32>
  return
}

func.func @print_f16(%v : vector<8xf16>) {
  %bits = arith.bitcast %v : vector<8xf16> to vector<8xi16>
  %u = vector.bitcast %bits : vector<8xi16> to vector<8xui16>
  vector.print %u : vector<8xui16>
  return
}

func.func @print_bf16(%v : vector<8xbf16>) {
  %bits = arith.bitcast %v : vector<8xbf16> to vector<8xi16>
  %u = vector.bitcast %bits : vector<8xi16> to vector<8xui16>
  vector.print %u : vector<8xui16>
  return
}

func.func @print_f64(%v : vector<8xf64>) {
  %bits = arith.bitcast %v : vector<8xf64> to vector<8xi64>
  %u = vector.bitcast %bits : vector<8xi64> to vector<8xui64>
  vector.print %u : vector<8xui64>
  return
}

func.func @extf_f8E4M3(%codes : vector<8xi8>) {
  %x = arith.bitcast %codes : vector<8xi8> to vector<8xf8E4M3>
  %0 = arith.extf %x : vector<8xf8E4M3> to vector<8xf32>
  func.call @print_f32(%0) : (vector<8xf32>) -> ()
  %1 = arith.extf %x : vector<8xf8E4M3> to vector<8xf16>
  func.call @print_f16(%1) : (vector<8xf16>) -> ()
  %2 = arith.extf %x : vector<8xf8E4M3> to vector<8xbf16>
  func.call @print_bf16(%2) : (vector<8xbf16>) -> ()
  %3 = arith.extf %x : vector<8xf8E4M3> to vector<8xf64>
  func.call @print_f64(%3) : (vector<8xf64>) -> ()
  return
}

func.func @extf_f8E3M4(%codes : vector<8xi8>) {
  %x = arith.bitcast %codes : vector<8xi8> to vector<8xf8E3M4>
  %0 = arith.extf %x : vector<8xf8E3M4> to vector<8xf32>
  func.call @print_f32(%0) : (vector<8xf32>) -> ()
  %1 = arith.extf %x : vector<8xf8E3M4> to vector<8xf16>
  func.call @print_f16(%1) : (vector<8xf16>) -> ()
  %2 = arith.extf %x : vector<8xf8E3M4> to vector<8xbf16>
  func.call @print_bf16(%2) : (vector<8xbf16>) -> ()
  %3 = arith.extf %x : vector<8xf8E3M4> to vector<8xf64>
  func.call @print_f64(%3) : (vector<8xf64>) -> ()
  return
}

func.func @extf_f8E4M3FNUZ(%codes : vector<8xi8>) {
  %x = arith.bitcast %codes : vector<8xi8> to vector<8xf8E4M3FNUZ>
  %0 = arith.extf %x : vector<8xf8E4M3FNUZ> to vector<8xf32>
  func.call @print_f32(%0) : (vector<8xf32>) -> ()
  %1 = arith.extf %x : vector<8xf8E4M3FNUZ> to vector<8xf16>
  func.call @print_f16(%1) : (vector<8xf16>) -> ()
  %2 = arith.extf %x : vector<8xf8E4M3FNUZ> to vector<8xbf16>
  func.call @print_bf16(%2) : (vector<8xbf16>) -> ()
  %3 = arith.extf %x : vector<8xf8E4M3FNUZ> to vector<8xf64>
  func.call @print_f64(%3) : (vector<8xf64>) -> ()
  return
}

func.func @extf_f8E5M2FNUZ(%codes : vector<8xi8>) {
  %x = arith.bitcast %codes : vector<8xi8> to vector<8xf8E5M2FNUZ>
  %0 = arith.extf %x : vector<8xf8E5M2FNUZ> to vector<8xf32>
  func.call @print_f32(%0) : (vector<8xf32>) -> ()
  %1 = arith.extf %x : vector<8xf8E5M2FNUZ> to vector<8xf16>
  func.call @print_f16(%1) : (vector<8xf16>) -> ()
  %2 = arith.extf %x : vector<8xf8E5M2FNUZ> to vector<8xbf16>
  func.call @print_bf16(%2) : (vector<8xbf16>) -> ()
  %3 = arith.extf %x : vector<8xf8E5M2FNUZ> to vector<8xf64>
  func.call @print_f64(%3) : (vector<8xf64>) -> ()
  return
}

func.func @extf_f8E4M3B11FNUZ(%codes : vector<8xi8>) {
  %x = arith.bitcast %codes : vector<8xi8> to vector<8xf8E4M3B11FNUZ>
  %0 = arith.extf %x : vector<8xf8E4M3B11FNUZ> to vector<8xf32>
  func.call @print_f32(%0) : (vector<8xf32>) -> ()
  %1 = arith.extf %x : vector<8xf8E4M3B11FNUZ> to vector<8xf16>
  func.call @print_f16(%1) : (vector<8xf16>) -> ()
  %2 = arith.extf %x : vector<8xf8E4M3B11FNUZ> to vector<8xbf16>
  func.call @print_bf16(%2) : (vector<8xbf16>) -> ()
  %3 = arith.extf %x : vector<8xf8E4M3B11FNUZ> to vector<8xf64>
  func.call @print_f64(%3) : (vector<8xf64>) -> ()
  return
}

func.func @extf_f6E2M3FN(%codes : vector<8xi8>) {
  %narrow = arith.trunci %codes : vector<8xi8> to vector<8xi6>
  %x = arith.bitcast %narrow : vector<8xi6> to vector<8xf6E2M3FN>
  %0 = arith.extf %x : vector<8xf6E2M3FN> to vector<8xf32>
  func.call @print_f32(%0) : (vector<8xf32>) -> ()
  %1 = arith.extf %x : vector<8xf6E2M3FN> to vector<8xf16>
  func.call @print_f16(%1) : (vector<8xf16>) -> ()
  %2 = arith.extf %x : vector<8xf6E2M3FN> to vector<8xbf16>
  func.call @print_bf16(%2) : (vector<8xbf16>) -> ()
  %3 = arith.extf %x : vector<8xf6E2M3FN> to vector<8xf64>
  func.call @print_f64(%3) : (vector<8xf64>) -> ()
  return
}

func.func @extf_f6E3M2FN(%codes : vector<8xi8>) {
  %narrow = arith.trunci %codes : vector<8xi8> to vector<8xi6>
  %x = arith.bitcast %narrow : vector<8xi6> to vector<8xf6E3M2FN>
  %0 = arith.extf %x : vector<8xf6E3M2FN> to vector<8xf32>
  func.call @print_f32(%0) : (vector<8xf32>) -> ()
  %1 = arith.extf %x : vector<8xf6E3M2FN> to vector<8xf16>
  func.call @print_f16(%1) : (vector<8xf16>) -> ()
  %2 = arith.extf %x : vector<8xf6E3M2FN> to vector<8xbf16>
  func.call @print_bf16(%2) : (vector<8xbf16>) -> ()
  %3 = arith.extf %x : vector<8xf6E3M2FN> to vector<8xf64>
  func.call @print_f64(%3) : (vector<8xf64>) -> ()
  return
}

func.func @entry() {
  // extf from each format; each @extf_* prints the f32, f16, bf16 and f64
  // results.

  // f8E4M3: 2^-9, 2^-6, 240, -240, +Inf, -Inf, sNaN 0x79, -qNaN 0xfc.
  %from_f8E4M3 = arith.constant dense<[
      0x01, 0x08, 0x77, 0xF7, 0x78, 0xF8, 0x79, 0xFC]> : vector<8xi8>
  // CHECK:      ( 989855744, 1015021568, 1131413504, 3278897152, 2139095040, 4286578688, 2144337920, 4290772992 )
  // CHECK-NEXT: ( 6144, 9216, 23424, 56192, 31744, 64512, 32384, 65024 )
  // CHECK-NEXT: ( 15104, 15488, 17264, 50032, 32640, 65408, 32720, 65472 )
  // CHECK-NEXT: ( 4566650022153682944, 4580160821035794432, 4642648265865560064, 13866020302720335872, 9218868437227405312, 18442240474082181120, 9221683186994511872, 18444492273895866368 )
  func.call @extf_f8E4M3(%from_f8E4M3) : (vector<8xi8>) -> ()

  // f8E3M4: 2^-6, 2^-2, 15.5, -0, -Inf, +Inf, sNaN 0x71, -qNaN 0xf9.
  %from_f8E3M4 = arith.constant dense<[
      0x01, 0x10, 0x6F, 0x80, 0xF0, 0x70, 0x71, 0xF9]> : vector<8xi8>
  // CHECK-NEXT: ( 1015021568, 1048576000, 1098383360, 2147483648, 4286578688, 2139095040, 2143813632, 4291297280 )
  // CHECK-NEXT: ( 9216, 13312, 19392, 32768, 64512, 31744, 32320, 65088 )
  // CHECK-NEXT: ( 15488, 16000, 16760, 32768, 65408, 32640, 32712, 65480 )
  // CHECK-NEXT: ( 4580160821035794432, 4598175219545276416, 4624915342332788736, 9223372036854775808, 18442240474082181120, 9218868437227405312, 9221401712017801216, 18444773748872577024 )
  func.call @extf_f8E3M4(%from_f8E3M4) : (vector<8xi8>) -> ()

  // f8E4M3FNUZ: 2^-10, 2^-7, 1, 240, -240, 0, NaN, -1.5.
  %from_f8E4M3FNUZ = arith.constant dense<[
      0x01, 0x08, 0x40, 0x7F, 0xFF, 0x00, 0x80, 0xC4]> : vector<8xi8>
  // CHECK-NEXT: ( 981467136, 1006632960, 1065353216, 1131413504, 3278897152, 0, 2143289344, 3217031168 )
  // CHECK-NEXT: ( 5120, 8192, 15360, 23424, 56192, 0, 32256, 48640 )
  // CHECK-NEXT: ( 14976, 15360, 16256, 17264, 50032, 0, 32704, 49088 )
  // CHECK-NEXT: ( 4562146422526312448, 4575657221408423936, 4607182418800017408, 4642648265865560064, 13866020302720335872, 0, 9221120237041090560, 13832806255468478464 )
  func.call @extf_f8E4M3FNUZ(%from_f8E4M3FNUZ) : (vector<8xi8>) -> ()

  // f8E5M2FNUZ: 2^-17, 3*2^-17, 2^-15, 57344, -57344, 0, NaN, -0.625.
  %from_f8E5M2FNUZ = arith.constant dense<[
      0x01, 0x03, 0x04, 0x7F, 0xFF, 0x00, 0x80, 0xBD]> : vector<8xi8>
  // CHECK-NEXT: ( 922746880, 935329792, 939524096, 1197473792, 3344957440, 0, 2143289344, 3206545408 )
  // CHECK-NEXT: ( 128, 384, 512, 31488, 64256, 0, 32256, 47360 )
  // CHECK-NEXT: ( 14080, 14272, 14336, 18272, 51040, 0, 32704, 48928 )
  // CHECK-NEXT: ( 4530621225134718976, 4537376624575774720, 4539628424389459968, 4678114112931102720, 13901486149785878528, 0, 9221120237041090560, 13827176755934265344 )
  func.call @extf_f8E5M2FNUZ(%from_f8E5M2FNUZ) : (vector<8xi8>) -> ()

  // f8E4M3B11FNUZ: 2^-13, 2^-10, 1, 30, -30, 0, NaN, -1.125.
  %from_f8E4M3B11FNUZ = arith.constant dense<[
      0x01, 0x08, 0x58, 0x7F, 0xFF, 0x00, 0x80, 0xD9]> : vector<8xi8>
  // CHECK-NEXT: ( 956301312, 981467136, 1065353216, 1106247680, 3253731328, 0, 2143289344, 3213885440 )
  // CHECK-NEXT: ( 2048, 5120, 15360, 20352, 53120, 0, 32256, 48256 )
  // CHECK-NEXT: ( 14592, 14976, 16256, 16880, 49648, 0, 32704, 49040 )
  // CHECK-NEXT: ( 4548635623644200960, 4562146422526312448, 4607182418800017408, 4629137466983448576, 13852509503838224384, 0, 9221120237041090560, 13831117405608214528 )
  func.call @extf_f8E4M3B11FNUZ(%from_f8E4M3B11FNUZ) : (vector<8xi8>) -> ()

  // f6E2M3FN: 0.125, 0.875, 1, 7.5, -7.5, -0, -0.125, 2.75.
  %from_f6E2M3FN = arith.constant dense<[
      0x01, 0x07, 0x08, 0x1F, 0x3F, 0x20, 0x21, 0x13]> : vector<8xi8>
  // CHECK-NEXT: ( 1040187392, 1063256064, 1065353216, 1089470464, 3236954112, 2147483648, 3187671040, 1076887552 )
  // CHECK-NEXT: ( 12288, 15104, 15360, 18304, 51072, 32768, 45056, 16768 )
  // CHECK-NEXT: ( 15872, 16224, 16256, 16624, 49392, 32768, 48640, 16432 )
  // CHECK-NEXT: ( 4593671619917905920, 4606056518893174784, 4607182418800017408, 4620130267728707584, 13843502304583483392, 9223372036854775808, 13817043656772681728, 4613374868287651840 )
  func.call @extf_f6E2M3FN(%from_f6E2M3FN) : (vector<8xi8>) -> ()

  // f6E3M2FN: 0.0625, 0.1875, 0.25, 28, -28, -0, -0.1875, 1.25.
  %from_f6E3M2FN = arith.constant dense<[
      0x01, 0x03, 0x04, 0x1F, 0x3F, 0x20, 0x23, 0x0D]> : vector<8xi8>
  // CHECK-NEXT: ( 1031798784, 1044381696, 1048576000, 1105199104, 3252682752, 2147483648, 3191865344, 1067450368 )
  // CHECK-NEXT: ( 11264, 12800, 13312, 20224, 52992, 32768, 45568, 15616 )
  // CHECK-NEXT: ( 15744, 15936, 16000, 16864, 49632, 32768, 48704, 16288 )
  // CHECK-NEXT: ( 4589168020290535424, 4595923419731591168, 4598175219545276416, 4628574517030027264, 13851946553884803072, 9223372036854775808, 13819295456586366976, 4608308318706860032 )
  func.call @extf_f6E3M2FN(%from_f6E3M2FN) : (vector<8xi8>) -> ()
  return
}
