// RUN: mlir-opt -split-input-file -verify-diagnostics %s

func.func @invalid_cast(%a: index) {
  // expected-error @below {{cast incompatible}}
  %0 = index.casts %a : index to index
  return
}

// -----

func.func @invalid_cast(%a: i64) {
  // expected-error @below {{cast incompatible}}
  %0 = index.casts %a : i64 to i64
  return
}

// -----

func.func @invalid_cast(%a: index) {
  // expected-error @below {{cast incompatible}}
  %0 = index.castu %a : index to index
  return
}

// -----

func.func @invalid_cast(%a: i64) {
  // expected-error @below {{cast incompatible}}
  %0 = index.castu %a : i64 to i64
  return
}
