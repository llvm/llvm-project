// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// -----
// Unrecognized token: missing storage type maximum
// expected-error@+1 {{expected ':'}}
!qalias = !quant.any<i8<16>:f32>

// -----
// Unrecognized token: missing closing angle bracket
// expected-error@+1 {{expected '>'}}
!qalias = !quant<"any<i8<-4:3:f32>">

// -----
// Unrecognized token: missing type colon
// expected-error@+1 {{expected '>'}}
!qalias = !quant.any<i8<-4:3>f32>

// -----
// Unrecognized storage type: illegal prefix
// expected-error@+1 {{illegal storage type prefix}}
!qalias = !quant.any<int8<-4:3>:f32>

// -----
// Unrecognized storage type: no width
// expected-error@+1 {{illegal storage type prefix}}
!qalias = !quant.any<i<-4:3>:f32>

// -----
// Unrecognized storage type: storage size > 32
// expected-error@+1 {{illegal storage type size: 33}}
!qalias = !quant.any<i33:f32>

// -----
// Unrecognized storage type: storage size < 0
// expected-error@+1 {{illegal storage type size: 1024}}
!qalias = !quant.any<i1024<-4:3>:f32>

// -----
// Unrecognized storage type: storage size
// expected-error@+1 {{invalid integer width}}
!qalias = !quant.any<i0123123123123123<-4:3>:f32>

// -----
// Illegal storage min/max: max - min < 0
// expected-error@+1 {{illegal storage min and storage max: (2:1)}}
!qalias = !quant.any<i8<2:1>:f32>

// -----
// Illegal storage min/max: max - min == 0
// expected-error@+1 {{illegal storage min and storage max: (1:1)}}
!qalias = !quant.any<i8<1:1>:f32>

// -----
// Illegal storage min/max: max > defaultMax
// expected-error@+1 {{illegal storage type maximum: 9}}
!qalias = !quant.any<i4<-1:9>:f32>

// -----
// Illegal storage min/max: min < defaultMin
// expected-error@+1 {{illegal storage type minimum: -9}}
!qalias = !quant.any<i4<-9:1>:f32>
