// RUN: mlir-opt -allow-unregistered-dialect -split-input-file -verify-diagnostics --merge-alloc %s

func.func @block() {
  %mref = memref.alloc() : memref<8 x f32>
  %mref2 = memref.alloc() : memref<8 x f32>
  // expected-error@+1 {{expecting RegionBranchOpInterface for merge-alloc}}
  "some.block"() ({
   ^bb0:
    "some.use"(%mref) : (memref<8 x f32>) -> ()
   }) : () -> ()
}
