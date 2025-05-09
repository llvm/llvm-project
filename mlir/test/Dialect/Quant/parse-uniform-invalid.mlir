// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// -----
// Invalid type.
// expected-error@+1 {{unknown quantized type foobar}}
!qalias = !quant.foobar

// -----
// Unrecognized token: illegal token
// expected-error@+1 {{unknown quantized type __}}
!qalias = !quant.__

// -----
// Unrecognized token: trailing
// expected-error@+1 {{expected '>'}}
!qalias = !quant.uniform<i8<-4:3>:f32, 0.99872:127 23>

// -----
// Unrecognized token: missing storage type maximum
// expected-error@+1 {{expected ':'}}
!qalias = !quant.uniform<i8<16>:f32, 0.99872:127>

// -----
// Unrecognized token: missing closing angle bracket
// expected-error@+1 {{unbalanced '<' character in pretty dialect name}}
!qalias = !quant<uniform<i8<-4:3:f32, 0.99872:127>>

// -----
// Unrecognized token: missing type colon
// expected-error@+1 {{expected ':'}}
!qalias = !quant.uniform<i8<-4:3>f32, 0.99872:127>

// -----
// Unrecognized token: missing comma
// expected-error@+1 {{expected ','}}
!qalias = !quant.uniform<i8<-4:3>:f32 0.99872:127>

// -----
// Unrecognized storage type: illegal prefix
// expected-error@+1 {{illegal storage type prefix}}
!qalias = !quant.uniform<int8<-4:3>:f32, 0.99872:127>

// -----
// Unrecognized storage type: no width
// expected-error@+1 {{illegal storage type prefix}}
!qalias = !quant.uniform<i<-4:3>:f32, 0.99872:127>

// -----
// Unrecognized storage type: storage size > 32
// expected-error@+1 {{illegal storage type size: 33}}
!qalias = !quant.uniform<i33:f32, 0.99872:127>

// -----
// Unrecognized storage type: storage size < 0
// expected-error@+1 {{illegal storage type prefix}}
!qalias = !quant.uniform<i-1<-4:3>:f32, 0.99872:127>

// -----
// Unrecognized storage type: storage size
// expected-error@+1 {{invalid integer width}}
!qalias = !quant.uniform<i123123123120<-4:3>:f32, 0.99872:127>

// -----
// Illegal storage min/max: max - min < 0
// expected-error@+1 {{illegal storage min and storage max: (2:1)}}
!qalias = !quant.uniform<i8<2:1>:f32, 0.99872:127>

// -----
// Illegal storage min/max: max - min == 0
// expected-error@+1 {{illegal storage min and storage max: (1:1)}}
!qalias = !quant.uniform<i8<1:1>:f32, 0.99872:127>

// -----
// Illegal storage min/max: max > defaultMax
// expected-error@+1 {{illegal storage type maximum: 9}}
!qalias = !quant.uniform<i4<-1:9>:f32, 0.99872:127>

// -----
// Illegal storage min/max: min < defaultMin
// expected-error@+1 {{illegal storage type minimum: -9}}
!qalias = !quant.uniform<i4<-9:1>:f32, 0.99872:127>

// -----
// Illegal uniform params: invalid scale
// expected-error@+1 {{expected floating point literal}}
!qalias = !quant.uniform<i8<-4:3>:f32, abc:127>

// -----
// Illegal uniform params: invalid zero point separator
// expected-error@+1 {{expected '>'}}
!qalias = !quant.uniform<i8<-4:3>:f32, 0.1abc>

// -----
// Illegal uniform params: missing zero point
// expected-error@+1 {{expected integer value}}
!qalias = !quant.uniform<i8<-4:3>:f32, 0.1:>

// -----
// Illegal uniform params: invalid zero point
// expected-error@+1 {{expected integer value}}
!qalias = !quant.uniform<i8<-4:3>:f32, 0.1:abc>

// -----
// Illegal expressed type: f33
// expected-error@+1 {{expected non-function type}}
!qalias = !quant.uniform<i8<-4:3>:f33, 0.99872:127>

// -----
// Illegal scale: negative
// expected-error@+1 {{scale -1.000000e+00 out of expressed type range}}
!qalias = !quant.uniform<i8<-4:3>:f32, -1.0:127>

// -----
// Illegal uniform params: missing quantized dimension
// expected-error@+1 {{expected integer value}}
!qalias = !quant.uniform<i8<-4:3>:f32:, {2.000000e+02:-19.987200e-01:1}>

// -----
// Illegal uniform params: unspecified quantized dimension, when multiple scales
// provided.
// expected-error@+1 {{expected floating point literal}}
!qalias = !quant.uniform<i8<-4:3>:f32, {2.000000e+02,-19.987200e-01:1}>

// -----
// Illegal negative axis in per-axis quantization
// expected-error@+1 {{illegal quantized dimension: -1}}
!qalias = !quant.uniform<i8:f32:-1, {2.0,3.0:1}>

// -----
// Scale f16 underflow
// expected-error@+1 {{scale 5.800000e-08 out of expressed type range}}
!qalias = !quant.uniform<i8:f16, 5.8e-8>

// -----
// Scale f16 overflow
// expected-error@+1 {{scale 6.600000e+04 out of expressed type range}}
!qalias = !quant.uniform<i8:f16, 6.6e4>

// -----
// Scale f16 underflow in per-axis quantization
// expected-error@+1 {{scale 5.800000e-08 out of expressed type range}}
!qalias = !quant.uniform<i8:f16:1, {2.0,5.8e-8}>

// -----
// Scale f16 overflow in per-axis quantization
// expected-error@+1 {{scale 6.600000e+04 out of expressed type range}}
!qalias = !quant.uniform<i8:f16:1, {2.0,6.6e4}>

// -----
// Illegal negative axis in sub-channel quantization
// expected-error@+1 {{illegal quantized dimension: -1}}
!qalias = !quant.uniform<u8:f32:{0:1,-1:2},
    {{2.000000e+02:120,9.987200e-01:127}, {2.000000e+02,9.987200e-01}}>

// -----
// Illegal zero block-size in sub-channel quantization
// expected-error@+1 {{illegal block size: 0}}
!qalias = !quant.uniform<u8:f32:{0:0,1:2},
    {{2.000000e+02:120,9.987200e-01:127}, {2.000000e+02,9.987200e-01}}>

// -----
// Illegal negative block-size in sub-channel quantization
// expected-error@+1 {{illegal block size: -1}}
!qalias = !quant.uniform<u8:f32:{0:-1,1:2},
    {{2.000000e+02:120,9.987200e-01:127}, {2.000000e+02,9.987200e-01}}>

// -----
// Missing block size in sub-channel quantization
// expected-error@+1 {{expected ':'}}
!qalias = !quant.uniform<u8:f32:{0,1:2},
    {{2.000000e+02:120,9.987200e-01:127}, {2.000000e+02,9.987200e-01}}>

// -----
// Missing quantization dimension in sub-channel quantization
// expected-error@+1 {{expected integer value}}
!qalias = !quant.uniform<u8:f32:{:1,1:2},
    {{2.000000e+02:120,9.987200e-01:127}, {2.000000e+02,9.987200e-01}}>

// -----
// Invalid tensor literal structure in sub-channel quantization
// expected-error@+2 {{expected '>'}}
!qalias = !quant.uniform<u8:f32:{0:1,1:2},
    {2.000000e+02:120,9.987200e-01:127}, {2.000000e+02,9.987200e-01}>

// -----
// Ragged tensor literal in sub-channel quantization
// expected-error@+2 {{ranks are not consistent between elements}}
!qalias = !quant.uniform<u8:f32:{0:1,1:2},
    {{2.000000e+02:120,9.987200e-01:127}, {2.000000e+02}}>

// -----
// Missing braces around block-size information in sub-channel quantization
// expected-error@+1 {{expected ','}}
!qalias = !quant.uniform<u8:f32:0:1,1:2,
    {{2.000000e+02:120,9.987200e-01:127}, {2.000000e+02,9.987200e-01}}>

// -----
// Missing right-brace around block-size information in sub-channel quantization
// expected-error@+1 {{unbalanced '{' character}}
!qalias = !quant.uniform<u8:f32:{0:1,1:2,
      {{2.000000e+02:120,9.987200e-01:127}, {2.000000e+02,9.987200e-01}}>

// -----
// Missing left-brace around block-size information in sub-channel quantization
// expected-error@+1 {{unbalanced '<' character}}
!qalias = !quant.uniform<u8:f32:0:1,1:2},
    {{2.000000e+02:120,9.987200e-01:127}, {2.000000e+02,9.987200e-01}}>

// -----
// Missing Axis:BlockSize pair
// expected-error@+1 {{expected integer value}}
!qalias = !quant.uniform<u8:f32:{0:1,},
    {{2.000000e+02:120,9.987200e-01:127}, {2.000000e+02,9.987200e-01}}>

// -----
// Missing Scale:ZeroPoint pair
// expected-error@+2 {{expected floating point literal}}
!qalias = !quant.uniform<u8:f32:{0:1,1:2},
    {{2.000000e+02:120,9.987200e-01:127}, {2.000000e+02,}}>

// -----
// Missing ZeroPoint in Scale:ZeroPoint pair
// expected-error@+2 {{expected integer value}}
!qalias = !quant.uniform<u8:f32:{0:1,1:2},
    {{2.000000e+02:120,9.987200e-01:127}, {2.000000e+02,9.987200e-01:}}>

// -----
// Empty quantization paramaters in sub-channel quantization
// expected-error@+1 {{expected floating point literal}}
!qalias = !quant.uniform<u8:f32:{0:1, 1:2}, {}>

// -----
// Scale out of expressed type range in sub-channel quantization
// expected-error@+2 {{scale 6.600000e+04 out of expressed type range}}
!qalias = !quant.uniform<i8:f16:{0:1,1:2},
    {{6.6e4:120,9.987200e-01:127}, {2.000000e+02:256,9.987200e-01}}>

