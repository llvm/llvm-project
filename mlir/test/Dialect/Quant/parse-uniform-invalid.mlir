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
// expected-error@+1 {{expected '>'}}
!qalias = !quant<"uniform<i8<-4:3:f32, 0.99872:127>">

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
// expected-error@+1 {{illegal scale: -1.000000}}
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
