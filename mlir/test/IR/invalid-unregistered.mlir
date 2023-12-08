// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// expected-error @below {{operation being parsed with an unregistered dialect}}
"unregistered_dialect.op"() : () -> ()

// -----

// expected-error @below {{attribute created with unregistered dialect}}
#attr = #unregistered_dialect.attribute

// -----

// expected-error @below {{type created with unregistered dialect}}
!type = !unregistered_dialect.type
