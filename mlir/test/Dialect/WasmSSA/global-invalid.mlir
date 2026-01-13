// RUN: mlir-opt %s -verify-diagnostics --split-input-file

module {
  // expected-error@+1 {{expected a constant initializer for this operator}}
  wasmssa.global @illegal i32 mutable : {
    %0 = wasmssa.const 17: i32
    %1 = wasmssa.const 35: i32
    %2 = wasmssa.add %0 %1 : i32
    wasmssa.return %2 : i32
  }
}

// -----

module {
  wasmssa.import_global "glob" from "my_module" as @global_0 mutable : i32
  wasmssa.global @global_1 i32 : {
  // expected-error@+1 {{global.get op is considered constant if it's referring to a import.global symbol marked non-mutable}}
    %0 = wasmssa.global_get @global_0 : i32
    wasmssa.return %0 : i32
  }
}

// -----

module {
  wasmssa.global @global_1 i32 : {
  // expected-error@+1 {{symbol @glarble is undefined}}
    %0 = wasmssa.global_get @glarble : i32
    wasmssa.return %0 : i32
  }
}

// -----

module {
  // expected-error@+1 {{expecting either `exported` or symbol name. got exproted}}
  wasmssa.global exproted @global_1 i32 : {
    %0 = wasmssa.const 17 : i32
    wasmssa.return %0 : i32
  }
}
