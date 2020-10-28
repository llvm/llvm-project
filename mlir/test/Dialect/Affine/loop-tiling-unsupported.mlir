// RUN: mlir-opt %s -affine-loop-tile="tile-size=32" -split-input-file -verify-diagnostics

// -----

#ub = affine_map<(d0)[s0] -> (d0, s0)>
func @non_hyperrect_loop() {
  %N = constant 128 : index
  // expected-error@+1 {{tiled code generation unimplemented for the non-hyperrectangular case}}
  affine.for %i = 0 to %N {
    affine.for %j = 0 to min #ub(%i)[%N] {
      affine.yield
    }
  }
  return
}
