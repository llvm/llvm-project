// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// expected-error @below {{op created with unregistered dialect}}
"unregistered_dialect.op"() : () -> ()

// -----

// expected-error @below {{attribute created with unregistered dialect}}
#attr = #unregistered_dialect.attribute

// -----

// expected-error @below {{type created with unregistered dialect}}
!type = type !unregistered_dialect.type
