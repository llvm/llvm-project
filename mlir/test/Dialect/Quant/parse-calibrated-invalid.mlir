// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// -----
// Unrecognized token: missing calibrated type maximum
// expected-error@+2 {{calibrated values must be present}}
// expected-error@+1 {{expected ':'}}
!qalias = !quant.calibrated<f32<-0.998>>

// -----
// Unrecognized token: missing closing angle bracket
// expected-error@+1 {{unbalanced '<' character in pretty dialect name}}
!qalias = !quant.calibrated<f32<-0.998:1.232>

// -----
// Unrecognized expressed type: integer type
// expected-error@+2 {{invalid kind of type specified}}
// expected-error@+1 {{expecting float expressed type}}
!qalias = !quant.calibrated<i8<-4:3>>

// -----
// Illegal storage min/max: max - min < 0
// expected-error@+1 {{illegal min and max: (1.000000e+00:-1.000000e+00)}}
!qalias = !quant.calibrated<f32<1.0:-1.0>>

// -----
// Illegal storage min/max: max - min == 0
// expected-error@+1 {{illegal min and max: (1.000000e+00:1.000000e+00)}}
!qalias = !quant.calibrated<f32<1.0:1.0>>
