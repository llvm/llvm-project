// RUN: mlir-opt -allow-unregistered-dialect %s -split-input-file -canonicalize="test-convergence" | FileCheck %s
// RUN: mlir-opt -allow-unregistered-dialect %s -split-input-file -canonicalize="test-convergence top-down=0" | FileCheck %s --check-prefix=CHECK-BOTTOM-UP

// -----

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0) -> (d0 - 1)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0) -> (d0 + 1)>

// CHECK-LABEL: func @compose_affine_maps_1dto2d_no_symbols() {
func.func @compose_affine_maps_1dto2d_no_symbols() {
  %0 = memref.alloc() : memref<4x4xf32>

  affine.for %i0 = 0 to 15 {
    // Test load[%x, %x]

    %x0 = affine.apply affine_map<(d0) -> (d0 - 1)> (%i0)
    %x1_0 = affine.apply affine_map<(d0, d1) -> (d0)> (%x0, %x0)
    %x1_1 = affine.apply affine_map<(d0, d1) -> (d1)> (%x0, %x0)

    // CHECK: %[[I0A:.*]] = affine.apply #[[$MAP0]](%{{.*}})
    // CHECK-NEXT: %[[V0:.*]] = memref.load %{{.*}}[%[[I0A]], %[[I0A]]]
    %v0 = memref.load %0[%x1_0, %x1_1] : memref<4x4xf32>

    // Test store[%y, %y]
    %y0 = affine.apply affine_map<(d0) -> (d0 + 1)> (%i0)
    %y1_0 = affine.apply affine_map<(d0, d1) -> (d0)> (%y0, %y0)
    %y1_1 = affine.apply affine_map<(d0, d1) -> (d1)> (%y0, %y0)

    // CHECK-NEXT: %[[I1A:.*]] = affine.apply #[[$MAP1]](%{{.*}})
    // CHECK-NEXT: memref.store %[[V0]], %{{.*}}[%[[I1A]], %[[I1A]]]
    memref.store %v0, %0[%y1_0, %y1_1] : memref<4x4xf32>

    // Test store[%x, %y]
    %xy_0 = affine.apply affine_map<(d0, d1) -> (d0)> (%x0, %y0)
    %xy_1 = affine.apply affine_map<(d0, d1) -> (d1)> (%x0, %y0)

    // CHECK-NEXT: memref.store %[[V0]], %{{.*}}[%[[I0A]], %[[I1A]]]
    memref.store %v0, %0[%xy_0, %xy_1] : memref<4x4xf32>

    // Test store[%y, %x]
    %yx_0 = affine.apply affine_map<(d0, d1) -> (d0)> (%y0, %x0)
    %yx_1 = affine.apply affine_map<(d0, d1) -> (d1)> (%y0, %x0)
    // CHECK-NEXT: memref.store %[[V0]], %{{.*}}[%[[I1A]], %[[I0A]]]
    memref.store %v0, %0[%yx_0, %yx_1] : memref<4x4xf32>
  }
  return
}

// -----

// CHECK-DAG: #[[$MAP4:.*]] = affine_map<(d0) -> (d0 - 4)>
// CHECK-DAG: #[[$MAP7:.*]] = affine_map<(d0) -> (d0 * 2 - 3)>
// CHECK-DAG: #[[$MAP7a:.*]] = affine_map<(d0) -> (d0 * 2 + 1)>

// CHECK-LABEL: func @compose_affine_maps_1dto2d_with_symbols() {
func.func @compose_affine_maps_1dto2d_with_symbols() {
  %0 = memref.alloc() : memref<4x4xf32>

  affine.for %i0 = 0 to 15 {
    // Test load[%x0, %x0] with symbol %c4
    %c4 = arith.constant 4 : index
    %x0 = affine.apply affine_map<(d0)[s0] -> (d0 - s0)> (%i0)[%c4]

    // CHECK: %[[I0:.*]] = affine.apply #[[$MAP4]](%{{.*}})
    // CHECK-NEXT: %[[V0:.*]] = memref.load %{{.*}}[%[[I0]], %[[I0]]]
    %v0 = memref.load %0[%x0, %x0] : memref<4x4xf32>

    // Test load[%x0, %x1] with symbol %c4 captured by '%x0' map.
    %x1 = affine.apply affine_map<(d0) -> (d0 + 1)> (%i0)
    %y1 = affine.apply affine_map<(d0, d1) -> (d0+d1)> (%x0, %x1)
    // CHECK-NEXT: %[[I1:.*]] = affine.apply #[[$MAP7]](%{{.*}})
    // CHECK-NEXT: memref.store %[[V0]], %{{.*}}[%[[I1]], %[[I1]]]
    memref.store %v0, %0[%y1, %y1] : memref<4x4xf32>

    // Test store[%x1, %x0] with symbol %c4 captured by '%x0' map.
    %y2 = affine.apply affine_map<(d0, d1) -> (d0 + d1)> (%x1, %x0)
    // CHECK-NEXT: %[[I2:.*]] = affine.apply #[[$MAP7]](%{{.*}})
    // CHECK-NEXT: memref.store %[[V0]], %{{.*}}[%[[I2]], %[[I2]]]
    memref.store %v0, %0[%y2, %y2] : memref<4x4xf32>

    // Test store[%x2, %x0] with symbol %c4 from '%x0' and %c5 from '%x2'
    %c5 = arith.constant 5 : index
    %x2 = affine.apply affine_map<(d0)[s0] -> (d0 + s0)> (%i0)[%c5]
    %y3 = affine.apply affine_map<(d0, d1) -> (d0 + d1)> (%x2, %x0)
    // CHECK: %[[I3:.*]] = affine.apply #[[$MAP7a]](%{{.*}})
    // CHECK-NEXT: memref.store %[[V0]], %{{.*}}[%[[I3]], %[[I3]]]
    memref.store %v0, %0[%y3, %y3] : memref<4x4xf32>
  }
  return
}

// -----

// CHECK-DAG: #[[$MAP8:.*]] = affine_map<(d0, d1) -> (d1 + (d0 ceildiv 4) * 4 - (d1 floordiv 4) * 4)>
// CHECK-DAG: #[[$MAP8a:.*]] = affine_map<(d0, d1) -> (d1 + (d0 ceildiv 8) * 8 - (d1 floordiv 8) * 8)>

// CHECK-LABEL: func @compose_affine_maps_2d_tile
func.func @compose_affine_maps_2d_tile(%0: memref<16x32xf32>, %1: memref<16x32xf32>) {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index

  affine.for %i0 = 0 to 16 {
    %x0 = affine.apply affine_map<(d0)[s0] -> (d0 ceildiv s0)> (%i0)[%c4]
    affine.for %i1 = 0 to 16 {
      %x1 = affine.apply affine_map<(d0)[s0] -> (d0 ceildiv s0)> (%i1)[%c8]
      affine.for %i2 = 0 to 16 {
        %x2 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)> (%i2)[%c4]
        affine.for %i3 = 0 to 16 {
          %x3 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)> (%i3)[%c8]

          %x40 = affine.apply affine_map<(d0, d1, d2, d3)[s0, s1] ->
            ((d0 * s0) + d2)> (%x0, %x1, %x2, %x3)[%c4, %c8]
          %x41 = affine.apply affine_map<(d0, d1, d2, d3)[s0, s1] ->
            ((d1 * s1) + d3)> (%x0, %x1, %x2, %x3)[%c4, %c8]
          // CHECK: %[[I0:.*]] = affine.apply #[[$MAP8]](%{{.*}}, %{{.*}})
          // CHECK: %[[I1:.*]] = affine.apply #[[$MAP8a]](%{{.*}}, %{{.*}})
          // CHECK-NEXT: %[[L0:.*]] = memref.load %{{.*}}[%[[I0]], %[[I1]]]
          %v0 = memref.load %0[%x40, %x41] : memref<16x32xf32>

          // CHECK-NEXT: memref.store %[[L0]], %{{.*}}[%[[I0]], %[[I1]]]
          memref.store %v0, %1[%x40, %x41] : memref<16x32xf32>
        }
      }
    }
  }
  return
}

// -----

// CHECK-DAG: #[[$MAP4b:.*]] = affine_map<(d0) -> (d0 - 7)>
// CHECK-DAG: #[[$MAP9:.*]] = affine_map<(d0) -> (d0 + 3)>
// CHECK-DAG: #[[$MAP10:.*]] = affine_map<(d0) -> (d0 * 3)>
// CHECK-DAG: #[[$MAP11:.*]] = affine_map<(d0) -> ((d0 + 3) ceildiv 3)>
// CHECK-DAG: #[[$MAP12:.*]] = affine_map<(d0) -> (d0 * 7 - 49)>

// CHECK-LABEL: func @compose_affine_maps_dependent_loads() {
func.func @compose_affine_maps_dependent_loads() {
  %0 = memref.alloc() : memref<16x32xf32>
  %1 = memref.alloc() : memref<16x32xf32>

  affine.for %i0 = 0 to 3 {
    affine.for %i1 = 0 to 3 {
      affine.for %i2 = 0 to 3 {
        %c3 = arith.constant 3 : index
        %c7 = arith.constant 7 : index

        %x00 = affine.apply affine_map<(d0, d1, d2)[s0, s1] -> (d0 + s0)>
            (%i0, %i1, %i2)[%c3, %c7]
        %x01 = affine.apply affine_map<(d0, d1, d2)[s0, s1] -> (d1 - s1)>
            (%i0, %i1, %i2)[%c3, %c7]
        %x02 = affine.apply affine_map<(d0, d1, d2)[s0, s1] -> (d2 * s0)>
            (%i0, %i1, %i2)[%c3, %c7]

        // CHECK: %[[I0:.*]] = affine.apply #[[$MAP9]](%{{.*}})
        // CHECK: %[[I1:.*]] = affine.apply #[[$MAP4b]](%{{.*}})
        // CHECK: %[[I2:.*]] = affine.apply #[[$MAP10]](%{{.*}})
        // CHECK-NEXT: %[[V0:.*]] = memref.load %{{.*}}[%[[I0]], %[[I1]]]
        %v0 = memref.load %0[%x00, %x01] : memref<16x32xf32>

        // CHECK-NEXT: memref.store %[[V0]], %{{.*}}[%[[I0]], %[[I2]]]
        memref.store %v0, %0[%x00, %x02] : memref<16x32xf32>

        // Swizzle %i0, %i1
        // CHECK-NEXT: memref.store %[[V0]], %{{.*}}[%[[I1]], %[[I0]]]
        memref.store %v0, %0[%x01, %x00] : memref<16x32xf32>

        // Swizzle %x00, %x01 and %c3, %c7
        %x10 = affine.apply affine_map<(d0, d1)[s0, s1] -> (d0 * s1)>
           (%x01, %x00)[%c3, %c7]
        %x11 = affine.apply affine_map<(d0, d1)[s0, s1] -> (d1 ceildiv s0)>
           (%x01, %x00)[%c3, %c7]

        // CHECK-NEXT: %[[I2A:.*]] = affine.apply #[[$MAP12]](%{{.*}})
        // CHECK-NEXT: %[[I2B:.*]] = affine.apply #[[$MAP11]](%{{.*}})
        // CHECK-NEXT: memref.store %[[V0]], %{{.*}}[%[[I2A]], %[[I2B]]]
        memref.store %v0, %0[%x10, %x11] : memref<16x32xf32>
      }
    }
  }
  return
}

// -----

// CHECK-DAG: #[[$MAP13A:.*]] = affine_map<(d0) -> ((d0 + 6) ceildiv 8)>
// CHECK-DAG: #[[$MAP13B:.*]] = affine_map<(d0) -> ((d0 * 4 - 4) floordiv 3)>

// CHECK-LABEL: func @compose_affine_maps_diamond_dependency
func.func @compose_affine_maps_diamond_dependency(%arg0: f32, %arg1: memref<4x4xf32>) {
  affine.for %i0 = 0 to 15 {
    %a = affine.apply affine_map<(d0) -> (d0 - 1)> (%i0)
    %b = affine.apply affine_map<(d0) -> (d0 + 7)> (%a)
    %c = affine.apply affine_map<(d0) -> (d0 * 4)> (%a)
    %d0 = affine.apply affine_map<(d0, d1) -> (d0 ceildiv 8)> (%b, %c)
    %d1 = affine.apply affine_map<(d0, d1) -> (d1 floordiv 3)> (%b, %c)
    // CHECK: %[[I0:.*]] = affine.apply #[[$MAP13A]](%{{.*}})
    // CHECK: %[[I1:.*]] = affine.apply #[[$MAP13B]](%{{.*}})
    // CHECK-NEXT: memref.store %arg0, %arg1[%[[I0]], %[[I1]]]
    memref.store %arg0, %arg1[%d0, %d1] : memref<4x4xf32>
  }

  return
}

// -----

// CHECK-DAG: #[[$MAP14:.*]] = affine_map<()[s0, s1] -> ((s0 * 4 + s1 * 4) floordiv s0)>

// CHECK-LABEL: func @compose_affine_maps_multiple_symbols
func.func @compose_affine_maps_multiple_symbols(%arg0: index, %arg1: index) -> index {
  %a = affine.apply affine_map<(d0)[s0] -> (s0 + d0)> (%arg0)[%arg1]
  %c = affine.apply affine_map<(d0) -> (d0 * 4)> (%a)
  %e = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)> (%c)[%arg1]
  // CHECK: [[I0:.*]] = affine.apply #[[$MAP14]]()[%{{.*}}, %{{.*}}]
  return %e : index
}

// -----

// CHECK-LABEL: func @arg_used_as_dim_and_symbol
func.func @arg_used_as_dim_and_symbol(%arg0: memref<100x100xf32>, %arg1: index, %arg2: f32) -> (memref<100x100xf32, 1>, memref<1xi32>) {
  %c9 = arith.constant 9 : index
  %1 = memref.alloc() : memref<100x100xf32, 1>
  %2 = memref.alloc() : memref<1xi32>
  affine.for %i0 = 0 to 100 {
    affine.for %i1 = 0 to 100 {
      %3 = affine.apply affine_map<(d0, d1)[s0, s1] -> (d1 + s0 + s1)>
        (%i0, %i1)[%arg1, %c9]
      %4 = affine.apply affine_map<(d0, d1, d3) -> (d3 - (d0 + d1))>
        (%arg1, %c9, %3)
      // CHECK: memref.store %arg2, %{{.*}}[%{{.*}}, %{{.*}}]
      memref.store %arg2, %1[%4, %arg1] : memref<100x100xf32, 1>
    }
  }
  return %1, %2 : memref<100x100xf32, 1>, memref<1xi32>
}

// -----

// CHECK-LABEL: func @trivial_maps
func.func @trivial_maps() {
  // CHECK-NOT: affine.apply

  %0 = memref.alloc() : memref<10xf32>
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  affine.for %i1 = 0 to 10 {
    %1 = affine.apply affine_map<()[s0] -> (s0)>()[%c0]
    memref.store %cst, %0[%1] : memref<10xf32>
    %2 = memref.load %0[%c0] : memref<10xf32>

    %3 = affine.apply affine_map<()[] -> (0)>()[]
    memref.store %cst, %0[%3] : memref<10xf32>
    memref.store %2, %0[%c0] : memref<10xf32>
  }
  return
}

// -----

// CHECK-DAG: #[[$MAP15:.*]] = affine_map<()[s0] -> (s0 - 42)>

// CHECK-LABEL: func @partial_fold_map
func.func @partial_fold_map(%arg1: index, %arg2: index) -> index {
  // TODO: Constant fold one index into affine.apply
  %c42 = arith.constant 42 : index
  %2 = affine.apply affine_map<(d0, d1) -> (d0 - d1)> (%arg1, %c42)
  // CHECK: [[X:.*]] = affine.apply #[[$MAP15]]()[%{{.*}}]
  return %2 : index
}

// -----

// CHECK-DAG: #[[$MAP_symbolic_composition_a:.*]] = affine_map<()[s0] -> (s0 * 512)>

// CHECK-LABEL: func @symbolic_composition_a(%{{.*}}: index, %{{.*}}: index) -> index {
func.func @symbolic_composition_a(%arg0: index, %arg1: index) -> index {
  %0 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg0)
  %1 = affine.apply affine_map<()[s0, s1] -> (8 * s0)>()[%0, %arg0]
  %2 = affine.apply affine_map<()[s0, s1] -> (16 * s1)>()[%arg1, %1]
  // CHECK: %{{.*}} = affine.apply #[[$MAP_symbolic_composition_a]]()[%{{.*}}]
  return %2 : index
}

// -----

// CHECK-DAG: #[[$MAP_symbolic_composition_b:.*]] = affine_map<()[s0] -> (s0 * 4)>

// CHECK-LABEL: func @symbolic_composition_b(%arg0: index, %arg1: index, %arg2: index, %arg3: index) -> index {
func.func @symbolic_composition_b(%arg0: index, %arg1: index, %arg2: index, %arg3: index) -> index {
  %0 = affine.apply affine_map<(d0) -> (d0)>(%arg0)
  %1 = affine.apply affine_map<()[s0, s1, s2, s3] -> (s0 + s1 + s2 + s3)>()[%0, %0, %0, %0]
  // CHECK: %{{.*}} = affine.apply #[[$MAP_symbolic_composition_b]]()[%{{.*}}]
  return %1 : index
}

// -----

// CHECK-DAG: #[[$MAP_symbolic_composition_c:.*]] = affine_map<()[s0, s1] -> (s0 * 3 + s1)>

// CHECK-LABEL: func @symbolic_composition_c(%arg0: index, %arg1: index, %arg2: index, %arg3: index) -> index {
func.func @symbolic_composition_c(%arg0: index, %arg1: index, %arg2: index, %arg3: index) -> index {
  %0 = affine.apply affine_map<(d0) -> (d0)>(%arg0)
  %1 = affine.apply affine_map<(d0) -> (d0)>(%arg1)
  %2 = affine.apply affine_map<()[s0, s1, s2, s3] -> (s0 + s1 + s2 + s3)>()[%0, %0, %0, %1]
  // CHECK: %{{.*}} = affine.apply #[[$MAP_symbolic_composition_c]]()[%{{.*}}, %{{.*}}]
  return %2 : index
}

// -----

// CHECK-DAG: #[[$MAP_symbolic_composition_d:.*]] = affine_map<()[s0, s1] -> (s0 * 3 + s1)>

// CHECK-LABEL: func @symbolic_composition_d(
//  CHECK-SAME:   %[[ARG0:[0-9a-zA-Z]+]]: index
//  CHECK-SAME:   %[[ARG1:[0-9a-zA-Z]+]]: index
func.func @symbolic_composition_d(%arg0: index, %arg1: index, %arg2: index, %arg3: index) -> index {
  %0 = affine.apply affine_map<(d0) -> (d0)>(%arg0)
  %1 = affine.apply affine_map<()[s0] -> (s0)>()[%arg1]
  %2 = affine.apply affine_map<()[s0, s1, s2, s3] -> (s0 + s1 + s2 + s3)>()[%0, %0, %0, %1]
  // CHECK: %{{.*}} = affine.apply #[[$MAP_symbolic_composition_d]]()[%[[ARG0]], %[[ARG1]]]
  return %2 : index
}

// -----

// CHECK-DAG: #[[$MAP_mix_dims_and_symbols_b:.*]] = affine_map<()[s0, s1] -> (s0 * 42 + s1 + 6)>

// CHECK-LABEL: func @mix_dims_and_symbols_b(%arg0: index, %arg1: index) -> index {
func.func @mix_dims_and_symbols_b(%arg0: index, %arg1: index) -> index {
  %a = affine.apply affine_map<(d0)[s0] -> (d0 - 1 + 42 * s0)> (%arg0)[%arg1]
  %b = affine.apply affine_map<(d0) -> (d0 + 7)> (%a)
  // CHECK: {{.*}} = affine.apply #[[$MAP_mix_dims_and_symbols_b]]()[%{{.*}}, %{{.*}}]

  return %b : index
}

// -----

// CHECK-DAG: #[[$MAP_mix_dims_and_symbols_c:.*]] = affine_map<()[s0, s1] -> (s0 * 168 + s1 * 4 - 4)>

// CHECK-LABEL: func @mix_dims_and_symbols_c(%arg0: index, %arg1: index) -> index {
func.func @mix_dims_and_symbols_c(%arg0: index, %arg1: index) -> index {
  %a = affine.apply affine_map<(d0)[s0] -> (d0 - 1 + 42 * s0)> (%arg0)[%arg1]
  %b = affine.apply affine_map<(d0) -> (d0 + 7)> (%a)
  %c = affine.apply affine_map<(d0) -> (d0 * 4)> (%a)
  // CHECK: {{.*}} = affine.apply #[[$MAP_mix_dims_and_symbols_c]]()[%{{.*}}, %{{.*}}]
  return %c : index
}

// -----

// CHECK-DAG: #[[$MAP_mix_dims_and_symbols_d:.*]] = affine_map<()[s0, s1] -> ((s0 * 42 + s1 + 6) ceildiv 8)>

// CHECK-LABEL: func @mix_dims_and_symbols_d(%arg0: index, %arg1: index) -> index {
func.func @mix_dims_and_symbols_d(%arg0: index, %arg1: index) -> index {
  %a = affine.apply affine_map<(d0)[s0] -> (d0 - 1 + 42 * s0)> (%arg0)[%arg1]
  %b = affine.apply affine_map<(d0) -> (d0 + 7)> (%a)
  %c = affine.apply affine_map<(d0) -> (d0 * 4)> (%a)
  %d = affine.apply affine_map<()[s0] -> (s0 ceildiv 8)> ()[%b]
  // CHECK: {{.*}} = affine.apply #[[$MAP_mix_dims_and_symbols_d]]()[%{{.*}}, %{{.*}}]
  return %d : index
}

// -----

// CHECK-DAG: #[[$MAP_mix_dims_and_symbols_e:.*]] = affine_map<()[s0, s1] -> ((s0 * 168 + s1 * 4 - 4) floordiv 3)>

// CHECK-LABEL: func @mix_dims_and_symbols_e(%arg0: index, %arg1: index) -> index {
func.func @mix_dims_and_symbols_e(%arg0: index, %arg1: index) -> index {
  %a = affine.apply affine_map<(d0)[s0] -> (d0 - 1 + 42 * s0)> (%arg0)[%arg1]
  %b = affine.apply affine_map<(d0) -> (d0 + 7)> (%a)
  %c = affine.apply affine_map<(d0) -> (d0 * 4)> (%a)
  %d = affine.apply affine_map<()[s0] -> (s0 ceildiv 8)> ()[%b]
  %e = affine.apply affine_map<(d0) -> (d0 floordiv 3)> (%c)
  // CHECK: {{.*}} = affine.apply #[[$MAP_mix_dims_and_symbols_e]]()[%{{.*}}, %{{.*}}]
  return %e : index
}

// -----

// CHECK-LABEL: func @mix_dims_and_symbols_f(%arg0: index, %arg1: index) -> index {
func.func @mix_dims_and_symbols_f(%arg0: index, %arg1: index) -> index {
  %a = affine.apply affine_map<(d0)[s0] -> (d0 - 1 + 42 * s0)> (%arg0)[%arg1]
  %b = affine.apply affine_map<(d0) -> (d0 + 7)> (%a)
  %c = affine.apply affine_map<(d0) -> (d0 * 4)> (%a)
  %d = affine.apply affine_map<()[s0] -> (s0 ceildiv 8)> ()[%b]
  %e = affine.apply affine_map<(d0) -> (d0 floordiv 3)> (%c)
  %f = affine.apply affine_map<(d0, d1)[s0, s1] -> (d0 - s1 +  d1 - s0)> (%d, %e)[%e, %d]
  // CHECK: {{.*}} = arith.constant 0 : index

  return %f : index
}

// -----

// CHECK-DAG: #[[$MAP_symbolic_composition_b:.*]] = affine_map<()[s0] -> (s0 * 4)>

// CHECK-LABEL: func @mix_dims_and_symbols_g(%arg0: index, %arg1: index) -> (index, index, index) {
func.func @mix_dims_and_symbols_g(%M: index, %N: index) -> (index, index, index) {
  %K = affine.apply affine_map<(d0) -> (4*d0)> (%M)
  %res1 = affine.apply affine_map<()[s0, s1] -> (4 * s0)>()[%N, %K]
  %res2 = affine.apply affine_map<()[s0, s1] -> (s1)>()[%N, %K]
  %res3 = affine.apply affine_map<()[s0, s1] -> (1024)>()[%N, %K]
  // CHECK-DAG: {{.*}} = arith.constant 1024 : index
  // CHECK-DAG: {{.*}} = affine.apply #[[$MAP_symbolic_composition_b]]()[%{{.*}}]
  // CHECK-DAG: {{.*}} = affine.apply #[[$MAP_symbolic_composition_b]]()[%{{.*}}]
  return %res1, %res2, %res3 : index, index, index
}

// -----

// CHECK-DAG: #[[$symbolic_semi_affine:.*]] = affine_map<(d0)[s0] -> (d0 floordiv (s0 + 1))>

// CHECK-LABEL: func @symbolic_semi_affine(%arg0: index, %arg1: index, %arg2: memref<?xf32>) {
func.func @symbolic_semi_affine(%M: index, %N: index, %A: memref<?xf32>) {
  %f1 = arith.constant 1.0 : f32
  affine.for %i0 = 1 to 100 {
    %1 = affine.apply affine_map<()[s0] -> (s0 + 1)> ()[%M]
    %2 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)> (%i0)[%1]
    // CHECK-DAG: {{.*}} = affine.apply #[[$symbolic_semi_affine]](%{{.*}})[%{{.*}}]
    memref.store %f1, %A[%2] : memref<?xf32>
  }
  return
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<()[s0] -> (0, s0)>
// CHECK: #[[$MAP1:.*]] = affine_map<()[s0] -> (100, s0)>

// CHECK-LABEL:  func @constant_fold_bounds(%arg0: index) {
func.func @constant_fold_bounds(%N : index) {
  // CHECK:      arith.constant 3 : index
  // CHECK-NEXT: "foo"() : () -> index
  %c9 = arith.constant 9 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = affine.apply affine_map<(d0, d1) -> (d0 + d1)> (%c1, %c2)
  %l = "foo"() : () -> index

  // CHECK:  affine.for %{{.*}} = 5 to 7 {
  affine.for %i = max affine_map<(d0, d1) -> (0, d0 + d1)> (%c2, %c3) to min affine_map<(d0, d1) -> (d0 - 2, 32*d1)> (%c9, %c1) {
    "foo"(%i, %c3) : (index, index) -> ()
  }

  // Bound takes a non-constant argument but can still be folded.
  // CHECK:  affine.for %{{.*}} = 1 to 7 {
  affine.for %j = max affine_map<(d0) -> (0, 1)> (%N) to min affine_map<(d0, d1) -> (7, 9)> (%N, %l) {
    "foo"(%j, %c3) : (index, index) -> ()
  }

  // None of the bounds can be folded.
  // CHECK: affine.for %{{.*}} = max #[[$MAP0]]()[%{{.*}}] to min #[[$MAP1]]()[%{{.*}}] {
  affine.for %k = max affine_map<()[s0] -> (0, s0)> ()[%l] to min affine_map<()[s0] -> (100, s0)> ()[%N] {
    "foo"(%k, %c3) : (index, index) -> ()
  }
  return
}

// -----

// CHECK-LABEL:  func @fold_empty_loops()
func.func @fold_empty_loops() -> index {
  %c0 = arith.constant 0 : index
  affine.for %i = 0 to 10 {
  }
  %res = affine.for %i = 0 to 10 iter_args(%arg = %c0) -> index {
    affine.yield %arg : index
  }
  // CHECK-NEXT: %[[zero:.*]] = arith.constant 0
  // CHECK-NEXT: return %[[zero]]
  return %res : index
}

// -----

// CHECK-LABEL:  func @fold_empty_loop()
func.func @fold_empty_loop() -> (index, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %res:2 = affine.for %i = 0 to 10 iter_args(%arg0 = %c0, %arg1 = %c1) -> (index, index) {
    affine.yield %c2, %arg1 : index, index
  }
  // CHECK-DAG: %[[one:.*]] = arith.constant 1
  // CHECK-DAG: %[[two:.*]] = arith.constant 2
  // CHECK-NEXT: return %[[two]], %[[one]]
  return %res#0, %res#1 : index, index
}

// -----

// CHECK-LABEL:  func @fold_empty_loops_trip_count_1()
func.func @fold_empty_loops_trip_count_1() -> (index, index, index, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %res1:2 = affine.for %i = 0 to 1 iter_args(%arg0 = %c2, %arg1 = %c0) -> (index, index) {
    affine.yield %c1, %arg0 : index, index
  }
  %res2:2 = affine.for %i = 0 to 2 step 3 iter_args(%arg0 = %c2, %arg1 = %c0) -> (index, index) {
    affine.yield %arg1, %arg0 : index, index
  }
  // CHECK-DAG: %[[zero:.*]] = arith.constant 0
  // CHECK-DAG: %[[one:.*]] = arith.constant 1
  // CHECK-DAG: %[[two:.*]] = arith.constant 2
  // CHECK-NEXT: return %[[one]], %[[two]], %[[zero]], %[[two]]
  return %res1#0, %res1#1, %res2#0, %res2#1 : index, index, index, index
}

// -----

// CHECK-LABEL:  func @fold_empty_loop_trip_count_0()
func.func @fold_empty_loop_trip_count_0() -> (index, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %res:2 = affine.for %i = 0 to 0 iter_args(%arg0 = %c2, %arg1 = %c0) -> (index, index) {
    affine.yield %c1, %arg0 : index, index
  }
  // CHECK-DAG: %[[zero:.*]] = arith.constant 0
  // CHECK-DAG: %[[two:.*]] = arith.constant 2
  // CHECK-NEXT: return %[[two]], %[[zero]]
  return %res#0, %res#1 : index, index
}

// -----

// CHECK-LABEL:  func @fold_empty_loop_trip_count_unknown
func.func @fold_empty_loop_trip_count_unknown(%in : index) -> (index, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %res:2 = affine.for %i = 0 to %in iter_args(%arg0 = %c0, %arg1 = %c1) -> (index, index) {
    affine.yield %arg0, %arg1 : index, index
  }
  // CHECK-DAG: %[[zero:.*]] = arith.constant 0
  // CHECK-DAG: %[[one:.*]] = arith.constant 1
  // CHECK-NEXT: return %[[zero]], %[[one]]
  return %res#0, %res#1 : index, index
}

// -----

// CHECK-LABEL:  func @empty_loops_not_folded_1
func.func @empty_loops_not_folded_1(%in : index) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK: affine.for
  %res = affine.for %i = 0 to %in iter_args(%arg = %c0) -> index {
    affine.yield %c1 : index
  }
  return %res : index
}

// -----

// CHECK-LABEL:  func @empty_loops_not_folded_2
func.func @empty_loops_not_folded_2(%in : index) -> (index, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK: affine.for
  %res:2 = affine.for %i = 0 to %in iter_args(%arg0 = %c0, %arg1 = %c1) -> (index, index) {
    affine.yield %arg1, %arg0 : index, index
  }
  return %res#0, %res#1 : index, index
}

// -----

// CHECK-LABEL:  func @empty_loops_not_folded_3
func.func @empty_loops_not_folded_3() -> (index, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK: affine.for
  %res:2 = affine.for %i = 0 to 10 iter_args(%arg0 = %c0, %arg1 = %c1) -> (index, index) {
    affine.yield %arg1, %arg0 : index, index
  }
  return %res#0, %res#1 : index, index
}

// -----

// CHECK-LABEL:  func @zero_iter_loop_not_folded
func.func @zero_iter_loop_not_folded() {
  %A = memref.alloc() : memref<4xf32>
  affine.for %i = 0 to 0 {
      %load = affine.load %A[%i] : memref<4xf32>
      affine.store %load, %A[%i] : memref<4xf32>
  }
  // CHECK:   affine.for {{.*}} = 0 to 0 {
  return
}

// -----

// CHECK-LABEL:  func @fold_zero_iter_loops
// CHECK-SAME: %[[ARG:.*]]: index
func.func @fold_zero_iter_loops(%in : index) -> index {
  %c1 = arith.constant 1 : index
  %res = affine.for %i = 0 to 0 iter_args(%loop_arg = %in) -> index {
    %yield = arith.addi %loop_arg, %c1 : index
    affine.yield %yield : index
  }
  // CHECK-NEXT: return %[[ARG]]
  return %res : index
}

// -----

// CHECK-DAG: #[[$SET:.*]] = affine_set<(d0, d1)[s0] : (d0 >= 0, -d0 + 1022 >= 0, d1 >= 0, -d1 + s0 - 2 >= 0)>

// CHECK-LABEL: func @canonicalize_affine_if
//  CHECK-SAME:   %[[M:[0-9a-zA-Z]*]]: index,
//  CHECK-SAME:   %[[N:[0-9a-zA-Z]*]]: index)
func.func @canonicalize_affine_if(%M : index, %N : index) {
  %c1022 = arith.constant 1022 : index
  // Drop unused operand %M, propagate %c1022, and promote %N to symbolic.
  affine.for %i = 0 to 1024 {
    affine.for %j = 0 to %N {
      // CHECK: affine.if #[[$SET]](%{{.*}}, %{{.*}})[%[[N]]]
      affine.if affine_set<(d0, d1, d2, d3)[s0] : (d1 >= 0, d0 - d1 >= 0, d2 >= 0, d3 - d2 - 2 >= 0)>
          (%c1022, %i, %j, %N)[%M] {
        "foo"() : () -> ()
      }
      "bar"() : () -> ()
    }
  }
  return
}

// -----

// CHECK-DAG: #[[$SET:.*]] = affine_set<(d0, d1)[s0] : (d0 - 1 >= 0, d1 - 1 == 0, -d0 + s0 + 10 >= 0)>

// CHECK-LABEL: func @canonicalize_affine_if_compose_apply
// CHECK-SAME:   %[[N:.*]]: index
func.func @canonicalize_affine_if_compose_apply(%N: index) {
  %M = affine.apply affine_map<()[s0] -> (s0 + 10)> ()[%N]
  // CHECK-NEXT: affine.for %[[I:.*]] =
  affine.for %i = 0 to 1024 {
    // CHECK-NEXT: affine.for %[[J:.*]] =
    affine.for %j = 0 to 100 {
      %j_ = affine.apply affine_map<(d0)[] -> (d0 + 1)> (%j)
      // CHECK-NEXT: affine.if #[[$SET]](%[[I]], %[[J]])[%[[N]]]
      affine.if affine_set<(d0, d1)[s0] : (d0 - 1 >= 0, d1 - 2 == 0, -d0 + s0 >= 0)>(%i, %j_)[%M] {
        "test.foo"() : ()->()
      }
    }
  }
  return
}

// -----

// CHECK-DAG: #[[$LBMAP:.*]] = affine_map<()[s0] -> (0, s0)>
// CHECK-DAG: #[[$UBMAP:.*]] = affine_map<()[s0] -> (1024, s0 * 2)>

// CHECK-LABEL: func @canonicalize_bounds
// CHECK-SAME: %[[M:.*]]: index,
// CHECK-SAME: %[[N:.*]]: index)
func.func @canonicalize_bounds(%M : index, %N : index) {
  %c0 = arith.constant 0 : index
  %c1024 = arith.constant 1024 : index
  // Drop unused operand %N, drop duplicate operand %M, propagate %c1024, and
  // promote %M to a symbolic one.
  // CHECK: affine.for %{{.*}} = 0 to min #[[$UBMAP]]()[%[[M]]]
  affine.for %i = 0 to min affine_map<(d0, d1, d2, d3) -> (d0, d1 + d2)> (%c1024, %M, %M, %N) {
    "foo"() : () -> ()
  }
  // Promote %M to symbolic position.
  // CHECK: affine.for %{{.*}} = 0 to #{{.*}}()[%[[M]]]
  affine.for %i = 0 to affine_map<(d0) -> (4 * d0)> (%M) {
    "foo"() : () -> ()
  }
  // Lower bound canonicalize.
  // CHECK: affine.for %{{.*}} = max #[[$LBMAP]]()[%[[N]]] to %[[M]]
  affine.for %i = max affine_map<(d0, d1) -> (d0, d1)> (%c0, %N) to %M {
    "foo"() : () -> ()
  }
  return
}

// -----

// Compose maps into affine load and store ops.

// CHECK-LABEL: @compose_into_affine_load_store
func.func @compose_into_affine_load_store(%A : memref<1024xf32>, %u : index) {
  // CHECK: affine.for %[[IV:.*]] = 0 to 1024
  affine.for %i = 0 to 1024 {
    // Make sure the unused operand (%u below) gets dropped as well.
    %idx = affine.apply affine_map<(d0, d1) -> (d0 + 1)> (%i, %u)
    %0 = affine.load %A[%idx] : memref<1024xf32>
    affine.store %0, %A[%idx] : memref<1024xf32>
    // CHECK-NEXT: affine.load %{{.*}}[%[[IV]] + 1]
    // CHECK-NEXT: affine.store %{{.*}}, %{{.*}}[%[[IV]] + 1]

    // Map remains the same, but operand changes on composition.
    %copy = affine.apply affine_map<(d0) -> (d0)> (%i)
    %1 = affine.load %A[%copy] : memref<1024xf32>
    "prevent.dce"(%1) : (f32) -> ()
    // CHECK-NEXT: affine.load %{{.*}}[%[[IV]]]
  }
  return
}

// -----

func.func @affine_min(%arg0 : index, %arg1 : index, %arg2 : index) {
  %c511 = arith.constant 511 : index
  %c1 = arith.constant 0 : index
  %0 = affine.min affine_map<(d0)[s0] -> (1000, d0 + 512, s0 + 1)> (%c1)[%c511]
  "op0"(%0) : (index) -> ()
  // CHECK:       %[[CST:.*]] = arith.constant 512 : index
  // CHECK-NEXT:  "op0"(%[[CST]]) : (index) -> ()
  // CHECK-NEXT:  return
  return
}

// -----

func.func @affine_min(%arg0 : index, %arg1 : index, %arg2 : index) {
  %c3 = arith.constant 3 : index
  %c20 = arith.constant 20 : index
  %0 = affine.min affine_map<(d0)[s0] -> (1000, d0 floordiv 4, (s0 mod 5) + 1)> (%c20)[%c3]
  "op0"(%0) : (index) -> ()
  // CHECK:       %[[CST:.*]] = arith.constant 4 : index
  // CHECK-NEXT:  "op0"(%[[CST]]) : (index) -> ()
  // CHECK-NEXT:  return
  return
}

// -----

func.func @affine_max(%arg0 : index, %arg1 : index, %arg2 : index) {
  %c511 = arith.constant 511 : index
  %c1 = arith.constant 0 : index
  %0 = affine.max affine_map<(d0)[s0] -> (1000, d0 + 512, s0 + 1)> (%c1)[%c511]
  "op0"(%0) : (index) -> ()
  // CHECK:       %[[CST:.*]] = arith.constant 1000 : index
  // CHECK-NEXT:  "op0"(%[[CST]]) : (index) -> ()
  // CHECK-NEXT:  return
  return
}

// -----

func.func @affine_max(%arg0 : index, %arg1 : index, %arg2 : index) {
  %c3 = arith.constant 3 : index
  %c20 = arith.constant 20 : index
  %0 = affine.max affine_map<(d0)[s0] -> (1000, d0 floordiv 4, (s0 mod 5) + 1)> (%c20)[%c3]
  "op0"(%0) : (index) -> ()
  // CHECK:       %[[CST:.*]] = arith.constant 1000 : index
  // CHECK-NEXT:  "op0"(%[[CST]]) : (index) -> ()
  // CHECK-NEXT:  return
  return
}

// -----

// CHECK: #[[$MAP:.*]] = affine_map<(d0, d1) -> (d1 - 2, d0)>

func.func @affine_min(%arg0: index) {
  affine.for %i = 0 to %arg0 {
    affine.for %j = 0 to %arg0 {
      %c2 = arith.constant 2 : index
      // CHECK: affine.min #[[$MAP]]
      %0 = affine.min affine_map<(d0,d1,d2)->(d0, d1 - d2)>(%i, %j, %c2)
      "consumer"(%0) : (index) -> ()
    }
  }
  return
}

// -----

// Reproducer for PR45031. This used to fold into an incorrect map because
// symbols were concatenated in the wrong order during map folding. Map
// composition places the symbols of the original map before those of the map
// it is composed with, e.g. A.compose(B) will first have all symbols of A,
// then all symbols of B.

#map1 = affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>
#map2 = affine_map<(d0)[s0] -> (1024, -d0 + s0)>

// CHECK: #[[$MAP:.*]] = affine_map<()[s0, s1] -> (1024, s0 - s1 * 1024)>

// CHECK: func @rep(%[[ARG0:.*]]: index, %[[ARG1:.*]]: index)
func.func @rep(%arg0 : index, %arg1 : index) -> index {
  // CHECK-NOT: arith.constant
  %c0 = arith.constant 0 : index
  %c1024 = arith.constant 1024 : index
  // CHECK-NOT: affine.apply
  %0 = affine.apply #map1(%arg0)[%c1024, %c0]

  // CHECK: affine.min #[[$MAP]]()[%[[ARG1]], %[[ARG0]]]
  %1 = affine.min #map2(%0)[%arg1]
  return %1 : index
}

// -----

// CHECK-DAG: #[[ub:.*]] = affine_map<()[s0] -> (s0 + 2)>

func.func @drop_duplicate_bounds(%N : index) {
  // affine.for %i = max #lb(%arg0) to min #ub(%arg0)
  affine.for %i = max affine_map<(d0) -> (d0, d0)>(%N) to min affine_map<(d0) -> (d0 + 2, d0 + 2)>(%N) {
    "foo"() : () -> ()
  }
  return
}

// -----

// Ensure affine.parallel bounds expressions are canonicalized.

#map3 = affine_map<(d0) -> (d0 * 5)>

// CHECK-LABEL: func @affine_parallel_const_bounds
func.func @affine_parallel_const_bounds() {
  %cst = arith.constant 1.0 : f32
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %0 = memref.alloc() : memref<4xf32>
  // CHECK: affine.parallel (%{{.*}}) = (0) to (4)
  affine.parallel (%i) = (%c0) to (%c0 + %c4) {
    %1 = affine.apply #map3(%i)
    // CHECK: affine.parallel (%{{.*}}) = (0) to (%{{.*}} * 5)
    affine.parallel (%j) = (%c0) to (%1) {
      affine.store %cst, %0[%j] : memref<4xf32>
    }
  }
  return
}

// -----

func.func @compose_affine_maps_div_symbol(%A : memref<i64>, %i0 : index, %i1 : index) {
  %0 = affine.apply affine_map<()[s0] -> (2 * s0)> ()[%i0]
  %1 = affine.apply affine_map<()[s0] -> (3 * s0)> ()[%i0]
  %2 = affine.apply affine_map<(d0)[s0, s1] -> (d0 mod s1 + s0 * s1 + s0 * 4)> (%i1)[%0, %1]
  %3 = arith.index_cast %2: index to i64
  memref.store %3, %A[]: memref<i64>
  affine.for %i2 = 0 to 3 {
    %4 = affine.apply affine_map<(d0)[s0, s1] -> (d0 ceildiv s1 + s0 + s0 * 3)> (%i2)[%0, %1]
    %5 = arith.index_cast %4: index to i64
    memref.store %5, %A[]: memref<i64>
  }
  return
}

// -----

// CHECK: #[[MAP:.+]] = affine_map<()[s0, s1] -> (s0 + s1, s0 * s1)>

// CHECK: func @deduplicate_affine_min_expressions
// CHECK-SAME: (%[[I0:.+]]: index, %[[I1:.+]]: index)
func.func @deduplicate_affine_min_expressions(%i0: index, %i1: index) -> index {
  // CHECK:  affine.min #[[MAP]]()[%[[I0]], %[[I1]]]
  %0 = affine.min affine_map<()[s0, s1] -> (s0 + s1, s0 * s1, s1 + s0, s0 * s1)> ()[%i0, %i1]
  return %0: index
}

// -----

// CHECK: #[[MAP:.+]] = affine_map<()[s0, s1] -> (s0 + s1, s0 * s1)>

// CHECK: func @deduplicate_affine_max_expressions
// CHECK-SAME: (%[[I0:.+]]: index, %[[I1:.+]]: index)
func.func @deduplicate_affine_max_expressions(%i0: index, %i1: index) -> index {
  // CHECK:  affine.max #[[MAP]]()[%[[I0]], %[[I1]]]
  %0 = affine.max affine_map<()[s0, s1] -> (s0 + s1, s0 * s1, s1 + s0, s0 * s1)> ()[%i0, %i1]
  return %0: index
}

// -----

// CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0, s1, s2] -> (-s1 + s2, 16, s0 * 3)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0, s1, s2] -> (-s0 + s1, -s2 + 5, 16)>

// CHECK: func @merge_affine_min_ops
// CHECK-SAME: (%[[I0:.+]]: index, %[[I1:.+]]: index, %[[I2:.+]]: index, %[[I3:.+]]: index)
func.func @merge_affine_min_ops(%i0: index, %i1: index, %i2: index, %i3: index) -> (index, index) {
  %0 = affine.min affine_map<(d0)[s0] -> (16, d0 - s0)> (%i0)[%i1]

 // CHECK: affine.min #[[MAP0]]()[%[[I2]], %[[I1]], %[[I0]]]
  %1 = affine.min affine_map<(d0)[s0] -> (3 * s0, d0)> (%0)[%i2] // Use as dim
 // CHECK: affine.min #[[MAP1]]()[%[[I1]], %[[I0]], %[[I3]]]
  %2 = affine.min affine_map<(d0)[s0] -> (s0, 5 - d0)> (%i3)[%0] // Use as symbol

  return %1, %2: index, index
}

// -----

// CHECK: #[[MAP:.+]] = affine_map<()[s0, s1, s2] -> (s2 + 8, s2 * 4, s1 + 16, s1 * 8, s0 + 7)>

// CHECK: func @merge_multiple_affine_min_ops
// CHECK-SAME: (%[[I0:.+]]: index, %[[I1:.+]]: index, %[[I2:.+]]: index)
func.func @merge_multiple_affine_min_ops(%i0: index, %i1: index, %i2: index) -> index {
  %0 = affine.min affine_map<()[s0] -> (s0 + 16, s0 * 8)> ()[%i0]
  %1 = affine.min affine_map<()[s0] -> (s0 + 8, s0 * 4)> ()[%i1]
  // CHECK: affine.min #[[MAP]]()[%[[I2]], %[[I0]], %[[I1]]]
  %2 = affine.min affine_map<()[s0, s1, s2] -> (s0, 7 + s1, s2)> ()[%0, %i2, %1]
  return %2: index
}

// -----

// CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0, s1] -> (s1 + 16, s1 * 8, s0 * 2)>

// CHECK: func @merge_multiple_uses_of_affine_min_ops
// CHECK-SAME: (%[[I0:.+]]: index, %[[I1:.+]]: index)
func.func @merge_multiple_uses_of_affine_min_ops(%i0: index, %i1: index) -> index {
  %0 = affine.min affine_map<()[s0] -> (s0 + 16, s0 * 8)> ()[%i0]
  // CHECK: affine.min #[[MAP]]()[%[[I1]], %[[I0]]]
  %2 = affine.min affine_map<()[s0, s1, s2] -> (s0, s1, s2 * 2)> ()[%0, %0, %i1]
  return %2: index
}

// -----

// CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 + 16, s0 * 8)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0, s1, s2] -> (s2 + 16, s2 * 8, s1 * 2, s0 + 1)>

// CHECK: func @merge_mixed_uses_of_affine_min_ops
// CHECK-SAME: (%[[I0:.+]]: index, %[[I1:.+]]: index)
func.func @merge_mixed_uses_of_affine_min_ops(%i0: index, %i1: index) -> index {
  // CHECK: %[[AFFINE:.+]] = affine.min #[[MAP0]]()[%[[I0]]]
  %0 = affine.min affine_map<()[s0] -> (s0 + 16, s0 * 8)> ()[%i0]
  // %0 is bound to a symbol that is both a standalone expression and a part
  // of other expressions.
  // CHECK: affine.min #[[MAP1]]()[%[[AFFINE]], %[[I1]], %[[I0]]]
  %2 = affine.min affine_map<()[s0, s1, s2] -> (s0, s1 + 1, s2 * 2)> ()[%0, %0, %i1]
  return %2: index
}

// -----

// CHECK-LABEL: func @dont_merge_affine_min_if_not_single_dim
func.func @dont_merge_affine_min_if_not_single_dim(%i0: index, %i1: index, %i2: index) -> index {
  // CHECK-COUNT-2: affine.min
  %0 = affine.min affine_map<()[s0] -> (s0 + 16, s0 * 8)> ()[%i0]
  %1 = affine.min affine_map<(d0)[s0] -> (s0 + 4, 7 + d0)> (%0)[%i2]
  return %1: index
}

// -----

// CHECK-LABEL: func @dont_merge_affine_min_if_not_single_sym
func.func @dont_merge_affine_min_if_not_single_sym(%i0: index, %i1: index, %i2: index) -> index {
  // CHECK-COUNT-2: affine.min
  %0 = affine.min affine_map<()[s0] -> (s0 + 16, s0 * 8)> ()[%i0]
  %1 = affine.min affine_map<()[s0, s1] -> (s0 + 4, 7 + s1)> ()[%0, %i2]
  return %1: index
}

// -----

// CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0, s1, s2] -> (-s1 + s2, 16, s0 * 3)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0, s1, s2] -> (-s0 + s1, -s2 + 5, 16)>

// CHECK: func @merge_affine_max_ops
// CHECK-SAME: (%[[I0:.+]]: index, %[[I1:.+]]: index, %[[I2:.+]]: index, %[[I3:.+]]: index)
func.func @merge_affine_max_ops(%i0: index, %i1: index, %i2: index, %i3: index) -> (index, index) {
  %0 = affine.max affine_map<(d0)[s0] -> (16, d0 - s0)> (%i0)[%i1]

 // CHECK: affine.max #[[MAP0]]()[%[[I2]], %[[I1]], %[[I0]]]
  %1 = affine.max affine_map<(d0)[s0] -> (3 * s0, d0)> (%0)[%i2] // Use as dim
 // CHECK: affine.max #[[MAP1]]()[%[[I1]], %[[I0]], %[[I3]]]
  %2 = affine.max affine_map<(d0)[s0] -> (s0, 5 - d0)> (%i3)[%0] // Use as symbol

  return %1, %2: index, index
}

// -----

// CHECK: #[[MAP:.+]] = affine_map<()[s0, s1, s2] -> (s2 + 8, s2 * 4, s1 + 16, s1 * 8, s0 + 7)>

// CHECK: func @merge_multiple_affine_max_ops
// CHECK-SAME: (%[[I0:.+]]: index, %[[I1:.+]]: index, %[[I2:.+]]: index)
func.func @merge_multiple_affine_max_ops(%i0: index, %i1: index, %i2: index) -> index {
  %0 = affine.max affine_map<()[s0] -> (s0 + 16, s0 * 8)> ()[%i0]
  %1 = affine.max affine_map<()[s0] -> (s0 + 8, s0 * 4)> ()[%i1]
  // CHECK: affine.max #[[MAP]]()[%[[I2]], %[[I0]], %[[I1]]]
  %2 = affine.max affine_map<()[s0, s1, s2] -> (s0, 7 + s1, s2)> ()[%0, %i2, %1]
  return %2: index
}

// -----

// CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0, s1] -> (s1 + 16, s1 * 8, s0 * 2)>

// CHECK: func @merge_multiple_uses_of_affine_max_ops
// CHECK-SAME: (%[[I0:.+]]: index, %[[I1:.+]]: index)
func.func @merge_multiple_uses_of_affine_max_ops(%i0: index, %i1: index) -> index {
  %0 = affine.max affine_map<()[s0] -> (s0 + 16, s0 * 8)> ()[%i0]
  // CHECK: affine.max #[[MAP]]()[%[[I1]], %[[I0]]]
  %2 = affine.max affine_map<()[s0, s1, s2] -> (s0, s1, s2 * 2)> ()[%0, %0, %i1]
  return %2: index
}

// -----

// CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 + 16, s0 * 8)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0, s1, s2] -> (s2 + 16, s2 * 8, s1 * 2, s0 + 1)>

// CHECK: func @merge_mixed_uses_of_affine_max_ops
// CHECK-SAME: (%[[I0:.+]]: index, %[[I1:.+]]: index)
func.func @merge_mixed_uses_of_affine_max_ops(%i0: index, %i1: index) -> index {
  // CHECK: %[[AFFINE:.+]] = affine.max #[[MAP0]]()[%[[I0]]]
  %0 = affine.max affine_map<()[s0] -> (s0 + 16, s0 * 8)> ()[%i0]
  // %0 is bound to a symbol that is both a standalone expression and a part
  // of other expressions.
  // CHECK: affine.max #[[MAP1]]()[%[[AFFINE]], %[[I1]], %[[I0]]]
  %2 = affine.max affine_map<()[s0, s1, s2] -> (s0, s1 + 1, s2 * 2)> ()[%0, %0, %i1]
  return %2: index
}

// -----

// CHECK-LABEL: func @dont_merge_affine_max_if_not_single_dim
func.func @dont_merge_affine_max_if_not_single_dim(%i0: index, %i1: index, %i2: index) -> index {
  // CHECK-COUNT-2: affine.max
  %0 = affine.max affine_map<()[s0] -> (s0 + 16, s0 * 8)> ()[%i0]
  %1 = affine.max affine_map<(d0)[s0] -> (s0 + 4, 7 + d0)> (%0)[%i2]
  return %1: index
}

// -----

// CHECK-LABEL: func @dont_merge_affine_max_if_not_single_sym
func.func @dont_merge_affine_max_if_not_single_sym(%i0: index, %i1: index, %i2: index) -> index {
  // CHECK-COUNT-2: affine.max
  %0 = affine.max affine_map<()[s0] -> (s0 + 16, s0 * 8)> ()[%i0]
  %1 = affine.max affine_map<()[s0, s1] -> (s0 + 4, 7 + s1)> ()[%0, %i2]
  return %1: index
}

// -----

// Ensure bounding maps of affine.for are composed.

// CHECK-DAG: #[[$MAP0]] = affine_map<()[s0] -> (s0 - 2)>
// CHECK-DAG: #[[$MAP1]] = affine_map<()[s0] -> (s0 + 2)>

// CHECK-LABEL: func @compose_affine_for_bounds
// CHECK-SAME:   %[[N:.*]]: index)
// CHECK: affine.for %{{.*}} = #[[$MAP0]]()[%[[N]]] to #[[$MAP1]]()[%[[N]]] {

func.func @compose_affine_for_bounds(%N: index) {
  %u = affine.apply affine_map<(d0) -> (d0 + 2)>(%N)
  %l = affine.apply affine_map<(d0) -> (d0 - 2)>(%N)
  affine.for %i = %l to %u {
    "foo"() : () -> ()
  }
  return
}

// -----

// Compose maps into affine.vector_load / affine.vector_store

// CHECK-LABEL: func @compose_into_affine_vector_load_vector_store
// CHECK: affine.for %[[IV:.*]] = 0 to 1024
// CHECK-NEXT: affine.vector_load %{{.*}}[%[[IV]] + 1]
// CHECK-NEXT: affine.vector_store %{{.*}}, %{{.*}}[%[[IV]] + 1]
// CHECK-NEXT: affine.vector_load %{{.*}}[%[[IV]]]
func.func @compose_into_affine_vector_load_vector_store(%A : memref<1024xf32>, %u : index) {
  affine.for %i = 0 to 1024 {
    // Make sure the unused operand (%u below) gets dropped as well.
    %idx = affine.apply affine_map<(d0, d1) -> (d0 + 1)> (%i, %u)
    %0 = affine.vector_load %A[%idx] : memref<1024xf32>, vector<8xf32>
    affine.vector_store %0, %A[%idx] : memref<1024xf32>, vector<8xf32>

    // Map remains the same, but operand changes on composition.
    %copy = affine.apply affine_map<(d0) -> (d0)> (%i)
    %1 = affine.vector_load %A[%copy] : memref<1024xf32>, vector<8xf32>
    "prevent.dce"(%1) : (vector<8xf32>) -> ()
  }
  return
}

// -----

// CHECK-LABEL: func @no_fold_of_store
//  CHECK:   %[[cst:.+]] = memref.cast %arg
//  CHECK:   affine.store %[[cst]]
func.func @no_fold_of_store(%arg : memref<32xi8>, %holder: memref<memref<?xi8>>) {
  %0 = memref.cast %arg : memref<32xi8> to memref<?xi8>
  affine.store %0, %holder[] : memref<memref<?xi8>>
  return
}

// -----

// CHECK-DAG: #[[$MAP0:.+]] = affine_map<()[s0] -> (s0 + 16)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<()[s0] -> (s0 * 4)>

// CHECK: func @canonicalize_single_min_max
// CHECK-SAME: (%[[I0:.+]]: index, %[[I1:.+]]: index)
func.func @canonicalize_single_min_max(%i0: index, %i1: index) -> (index, index) {
  // CHECK-NOT: affine.min
  // CHECK-NEXT: affine.apply #[[$MAP0]]()[%[[I0]]]
  %0 = affine.min affine_map<()[s0] -> (s0 + 16)> ()[%i0]

  // CHECK-NOT: affine.max
  // CHECK-NEXT: affine.apply #[[$MAP1]]()[%[[I1]]]
  %1 = affine.min affine_map<()[s0] -> (s0 * 4)> ()[%i1]

  return %0, %1: index, index
}

// -----

// CHECK: #[[$MAP:.+]] = affine_map<()[s0, s1] -> (32, s1 + 16, s0 + s1)>

// CHECK-LABEL: func @canonicalize_multi_min_max
// CHECK-SAME: (%[[I0:.+]]: index, %[[I1:.+]]: index)
func.func @canonicalize_multi_min_max(%i0: index, %i1: index) -> (index, index) {
  // CHECK-NEXT: affine.min #[[$MAP]]()[%[[I0]], %[[I1]]]
  %0 = affine.min affine_map<()[s0, s1] -> (s0 + s1, s1 + 16, 32)> ()[%i0, %i1]

  // CHECK-NEXT: affine.max #[[$MAP]]()[%[[I0]], %[[I1]]]
  %1 = affine.max affine_map<()[s0, s1] -> (s0 + s1, 32, s1 + 16)> ()[%i0, %i1]

  return %0, %1: index, index
}

// -----

module {
  memref.global "private" constant @__constant_1x5x1xf32 : memref<1x5x1xf32> = dense<[[[6.250000e-02], [2.500000e-01], [3.750000e-01], [2.500000e-01], [6.250000e-02]]]>
  memref.global "private" constant @__constant_32x64xf32 : memref<32x64xf32> = dense<0.000000e+00>
  // CHECK-LABEL: func @fold_const_init_global_memref
  func.func @fold_const_init_global_memref() -> (f32, f32) {
    %m = memref.get_global @__constant_1x5x1xf32 : memref<1x5x1xf32>
    %v0 = affine.load %m[0, 0, 0] : memref<1x5x1xf32>
    %v1 = affine.load %m[0, 1, 0] : memref<1x5x1xf32>
    return %v0, %v1 : f32, f32
    // CHECK-DAG: %[[C0:.*]] = arith.constant 6.250000e-02 : f32
    // CHECK-DAG: %[[C1:.*]] = arith.constant 2.500000e-01 : f32
    // CHECK-NEXT: return %[[C0]], %[[C1]]
  }

  // CHECK-LABEL: func @fold_const_splat_global
  func.func @fold_const_splat_global() -> memref<32x64xf32> {
    // CHECK-NEXT: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
    %m = memref.get_global @__constant_32x64xf32 : memref<32x64xf32>
    %s = memref.alloc() : memref<32x64xf32>
    affine.for %i = 0 to 32 {
      affine.for %j = 0 to 64 {
        %v = affine.load %m[%i, %j] : memref<32x64xf32>
        affine.store %v, %s[%i, %j] : memref<32x64xf32>
        // CHECK: affine.store %[[CST]], %{{.*}}
      }
    }
    return %s: memref<32x64xf32>
  }
}

// -----

// Simplification of maps exploiting operand info.

// CHECK: #[[$MAP_SIMPLER:.*]] = affine_map<(d0, d1) -> (((d0 + d1) mod 458313) floordiv 227)>

// CHECK-LABEL: func @simplify_with_operands
func.func @simplify_with_operands(%N: index, %A: memref<?x32xf32>) {
  // CHECK-NEXT: affine.for %[[I:.*]] = 0 to %{{.*}}
  affine.for %i = 0 to %N step 32 {
    // CHECK-NEXT: affine.for %[[II:.*]] = 0 to 32
    affine.for %ii = 0 to 32 {
      // %ii is less than 32 and %i divides 32.
      // CHECK: affine.load %{{.*}}[0, 0]
      %x = affine.load %A[%ii floordiv 32, %i mod 32] : memref<?x32xf32>
      "test.foo"(%x) : (f32) -> ()

      // %i is aligned at 32 boundary and %ii < 32.
      // CHECK: affine.load %{{.*}}[%[[I]] floordiv 32, %[[II]] mod 16]
      %a = affine.load %A[(%i + %ii) floordiv 32, (%i + %ii) mod 16] : memref<?x32xf32>
      "test.foo"(%a) : (f32) -> ()
      // CHECK: affine.load %{{.*}}[%[[I]] floordiv 64, (%[[I]] + %[[II]]) mod 64]
      %b = affine.load %A[(%i + %ii) floordiv 64, (%i + %ii) mod 64] : memref<?x32xf32>
      "test.foo"(%b) : (f32) -> ()
      // CHECK: affine.load %{{.*}}[(%[[I]] + %[[II]]) floordiv 16, %[[II]] mod 16]
      %c = affine.load %A[(%i + %ii) floordiv 16, (%i + %ii) mod 16] : memref<?x32xf32>
      "test.foo"(%c) : (f32) -> ()
    }
  }

  // Should not simplify.
  affine.for %i = -1 to 32 {
    // CHECK: affine.load %{{.*}}[%{{.*}} floordiv {{.*}}, %{{.*}} mod {{.*}}] :
    %x = affine.load %A[%i floordiv 32, %i mod 32] : memref<?x32xf32>
    "test.foo"(%x) : (f32) -> ()
  }

  affine.for %arg0 = 0 to %N step 128 {
    affine.for %arg4 = 0 to 32 step 32 {
      affine.for %arg5 = 0 to 128 {
        // CHECK: affine.apply #[[$MAP_SIMPLER]]
        %x = affine.apply affine_map<(d0, d1, d2) -> (((d0 + d2) mod 458313) floordiv 227 + d1 floordiv 256)>(%arg0, %arg4, %arg5)
        "test.foo"(%x) : (index) -> ()
      }
    }
  }

  return
}

// CHECK-LABEL: func @simplify_div_mod_with_operands
func.func @simplify_div_mod_with_operands(%N: index, %A: memref<64xf32>, %unknown: index) {
  // CHECK: affine.for %[[I:.*]] = 0 to 32
  %cst = arith.constant 1.0 : f32
  affine.for %i = 0 to 32 {
    // CHECK: affine.store %{{.*}}, %{{.*}}[0]
    affine.store %cst, %A[%i floordiv 32] : memref<64xf32>
    // CHECK: affine.store %{{.*}}, %{{.*}}[1]
    affine.store %cst, %A[(%i + 1) ceildiv 32] : memref<64xf32>
    // CHECK: affine.store %{{.*}}, %{{.*}}[%[[I]]]
    affine.store %cst, %A[%i mod 32] : memref<64xf32>
    // CHECK: affine.store %{{.*}}, %{{.*}}[0]
    affine.store %cst, %A[2 * %i floordiv 64] : memref<64xf32>
    // CHECK: affine.store %{{.*}}, %{{.*}}[0]
    affine.store %cst, %A[(%i mod 16) floordiv 16] : memref<64xf32>

    // The ones below can't be simplified.
    affine.store %cst, %A[%i floordiv 16] : memref<64xf32>
    affine.store %cst, %A[%i mod 16] : memref<64xf32>
    affine.store %cst, %A[(%i mod 16) floordiv 15] : memref<64xf32>
    affine.store %cst, %A[%i mod 31] : memref<64xf32>
    // CHECK:      affine.store %{{.*}}, %{{.*}}[%{{.*}} floordiv 16] : memref<64xf32>
    // CHECK-NEXT: affine.store %{{.*}}, %{{.*}}[%{{.*}} mod 16] : memref<64xf32>
    // CHECK-NEXT: affine.store %{{.*}}, %{{.*}}[(%{{.*}} mod 16) floordiv 15] : memref<64xf32>
    // CHECK-NEXT: affine.store %{{.*}}, %{{.*}}[%{{.*}} mod 31] : memref<64xf32>
  }

  affine.for %i = -8 to 32 {
    // Can't be simplified.
    // CHECK: affine.store %{{.*}}, %{{.*}}[%{{.*}} floordiv 32] : memref<64xf32>
    affine.store %cst, %A[%i floordiv 32] : memref<64xf32>
    // CHECK: affine.store %{{.*}}, %{{.*}}[%{{.*}} mod 32] : memref<64xf32>
    affine.store %cst, %A[%i mod 32] : memref<64xf32>
    // floordiv rounds toward -inf; (%i - 96) floordiv 64 will be -2.
    // CHECK: affine.store %{{.*}}, %{{.*}}[0] : memref<64xf32>
    affine.store %cst, %A[2 + (%i - 96) floordiv 64] : memref<64xf32>
  }

  // CHECK: affine.for %[[II:.*]] = 8 to 16
  affine.for %i = 8 to 16 {
    // CHECK: affine.store %{{.*}}, %{{.*}}[1] : memref<64xf32>
    affine.store %cst, %A[%i floordiv 8] : memref<64xf32>
    // CHECK: affine.store %{{.*}}, %{{.*}}[2] : memref<64xf32>
    affine.store %cst, %A[(%i + 1) ceildiv 8] : memref<64xf32>
    // CHECK: affine.store %{{.*}}, %{{.*}}[%[[II]] mod 8] : memref<64xf32>
    affine.store %cst, %A[%i mod 8] : memref<64xf32>
    // CHECK: affine.store %{{.*}}, %{{.*}}[%[[II]]] : memref<64xf32>
    affine.store %cst, %A[%i mod 32] : memref<64xf32>
    // Upper bound on the mod 32 expression will be 15.
    // CHECK: affine.store %{{.*}}, %{{.*}}[0] : memref<64xf32>
    affine.store %cst, %A[(%i mod 32) floordiv 16] : memref<64xf32>
    // Lower bound on the mod 16 expression will be 8.
    // CHECK: affine.store %{{.*}}, %{{.*}}[1] : memref<64xf32>
    affine.store %cst, %A[(%i mod 16) floordiv 8] : memref<64xf32>
    // CHECK: affine.store %{{.*}}, %{{.*}}[0] : memref<64xf32>
    affine.store %cst, %A[(%unknown mod 16) floordiv 16] : memref<64xf32>
  }
  return
}

// -----

#map0 = affine_map<(d0) -> (32, d0 * -32 + 32)>
#map1 = affine_map<(d0) -> (32, d0 * -32 + 64)>
#map3 = affine_map<(d0) -> (16, d0 * -16 + 32)>

// CHECK-DAG: #[[$SIMPLE_MAP:.*]] = affine_map<()[s0] -> (3, s0)>
// CHECK-DAG: #[[$SIMPLE_MAP_MAX:.*]] = affine_map<()[s0] -> (5, s0)>
// CHECK-DAG: #[[$SIMPLIFIED_MAP:.*]] = affine_map<(d0, d1) -> (-9, d0 * 4 - d1 * 4)>
// CHECK-DAG: #[[$FLOORDIV:.*]] = affine_map<(d0) -> (d0 floordiv 2)>

// CHECK-LABEL: func @simplify_min_max_bounds_simple
func.func @simplify_min_max_bounds_simple(%M: index) {

  // CHECK-NEXT: affine.for %{{.*}} = 0 to min #[[$SIMPLE_MAP]]
  affine.for %i = 0 to min affine_map<(d0) -> (3, 5, d0)>(%M) {
    "test.foo"() : () -> ()
  }

  // CHECK: affine.for %{{.*}} = 0 to min #[[$SIMPLE_MAP]]
  affine.for %i = 0 to min affine_map<(d0) -> (3, 3, d0)>(%M) {
    "test.foo"() : () -> ()
  }

  // CHECK: affine.for %{{.*}} = max #[[$SIMPLE_MAP_MAX]]
  affine.for %i = max affine_map<(d0) -> (3, 5, d0)>(%M) to 10 {
    "test.foo"() : () -> ()
  }

  // CHECK: affine.for %{{.*}} = max #[[$SIMPLE_MAP_MAX]]
  affine.for %i = max affine_map<(d0) -> (5, 5, d0)>(%M) to 10 {
    "test.foo"() : () -> ()
  }

  return
}

// CHECK-LABEL: func @simplify_bounds_tiled
func.func @simplify_bounds_tiled() {
  affine.for %arg5 = 0 to 1 {
    affine.for %arg6 = 0 to 2 {
      affine.for %arg8 = 0 to min #map0(%arg5) step 16 {
        affine.for %arg9 = 0 to min #map1(%arg6) step 16 {
          affine.for %arg10 = 0 to 2 {
            affine.for %arg12 = 0 to min #map3(%arg10) step 16 {
              "test.foo"() : () -> ()
            }
          }
        }
      }
    }
  }
  // CHECK:      affine.for
  // CHECK-NEXT:   affine.for
  // CHECK-NEXT:     affine.for %{{.*}} = 0 to 32 step 16
  // CHECK-NEXT:       affine.for %{{.*}} = 0 to 32 step 16
  // CHECK-NEXT:         affine.for %{{.*}} = 0 to 2
  // CHECK-NEXT:           affine.for %{{.*}} = 0 to 16 step 16

  return
}

// CHECK-LABEL: func @simplify_min_max_multi_expr
func.func @simplify_min_max_multi_expr() {
  // Lower bound max.
  // CHECK: affine.for
  affine.for %i = 0 to 2 {
    // CHECK: affine.for %{{.*}} = 5 to
    affine.for %j = max affine_map<(d0) -> (5, 4 * d0)> (%i) to affine_map<(d0) -> (4 * d0 + 3)>(%i) {
      "test.foo"() : () -> ()
    }
  }

  // Expressions with multiple operands.
  // CHECK: affine.for
  affine.for %i = 0 to 2 {
    // CHECK: affine.for
    affine.for %j = 0 to 4 {
      // The first upper bound expression will not be lower than -9. So, it's redundant.
      // CHECK-NEXT: affine.for %{{.*}} = -10 to -9
      affine.for %k = -10 to min affine_map<(d0, d1) -> (4 * d0 - 3 * d1, -9)>(%i, %j) {
        "test.foo"() : () -> ()
      }
    }
  }

  // One expression is redundant but not the others.
  // CHECK: affine.for
  affine.for %i = 0 to 2 {
    // CHECK: affine.for
    affine.for %j = 0 to 4 {
      // The first upper bound expression will not be lower than -9. So, it's redundant.
      // CHECK-NEXT: affine.for %{{.*}} = -10 to min #[[$SIMPLIFIED_MAP]]
      affine.for %k = -10 to min affine_map<(d0, d1) -> (4 * d0 - 3 * d1, -9, 4 * d0 - 4 * d1)>(%i, %j) {
        "test.foo"() : () -> ()
      }
    }
  }

  // CHECK: affine.for %{{.*}} = 0 to 1
  affine.for %i = 0 to 2 {
    affine.for %j = max affine_map<(d0) -> (d0 floordiv 2, 0)>(%i) to 1 {
      "test.foo"() : () -> ()
    }
  }

  // The constant bound is redundant here.
  // CHECK: affine.for %{{.*}} = #[[$FLOORDIV]](%{{.*}} to 10
  affine.for %i = 0 to 8 {
    affine.for %j = max affine_map<(d0) -> (d0 floordiv 2, 0)>(%i) to 10 {
      "test.foo"() : () -> ()
    }
  }

  return
}

// CHECK-LABEL: func @no_simplify_min_max
func.func @no_simplify_min_max(%M: index) {
  // Negative test cases.
  // CHECK: affine.for
  affine.for %i = 0 to 4 {
    // CHECK-NEXT: affine.for %{{.*}} = 0 to min
    affine.for %j = 0 to min affine_map<(d0) -> (2 * d0, 2)>(%i) {
      "test.foo"() : () -> ()
    }
    // CHECK:      affine.for %{{.*}} = 0 to min {{.*}}(%{{.*}})[%{{.*}}]
    affine.for %j = 0 to min affine_map<(d0)[s0] -> (d0, s0)>(%i)[%M] {
      "test.foo"() : () -> ()
    }
  }

  return
}

// -----

//           CHECK: #[[$map:.*]] = affine_map<()[s0] -> (s0 * ((-s0 + 40961) ceildiv 512))>
// CHECK-BOTTOM-UP: #[[$map:.*]] = affine_map<()[s0] -> (s0 * ((-s0 + 40961) ceildiv 512))>
//           CHECK-LABEL: func @regression_do_not_perform_invalid_replacements
// CHECK-BOTTOM-UP-LABEL: func @regression_do_not_perform_invalid_replacements
func.func @regression_do_not_perform_invalid_replacements(%arg0: index) {
  // Dim must be promoted to sym before combining both maps.
  //           CHECK: %[[apply:.*]] = affine.apply #[[$map]]()[%{{.*}}]
  // CHECK-BOTTOM-UP: %[[apply:.*]] = affine.apply #[[$map]]()[%{{.*}}]
  %0 = affine.apply affine_map<(d0) -> (-d0 + 40961)>(%arg0)
  %1 = affine.apply affine_map<(d0)[s0] -> (d0 * (s0 ceildiv 512))>(%arg0)[%0]
  //           CHECK: "test.foo"(%[[apply]])
  // CHECK-BOTTOM-UP: "test.foo"(%[[apply]])
  "test.foo"(%1) : (index) -> ()
  return
}

// -----
// CHECK-LABEL: func @min.oneval(%arg0: index)
func.func @min.oneval(%arg0: index) -> index {
  %min = affine.min affine_map<()[s0] -> (s0)> ()[%arg0]
  // CHECK: return %arg0 : index
  return %min: index
}

// -----
// CHECK-LABEL: func @max.oneval(%arg0: index)
func.func @max.oneval(%arg0: index) -> index {
  %max = affine.max affine_map<()[s0] -> (s0)> ()[%arg0]
  // CHECK: return %arg0 : index
  return %max: index
}

// -----

// CHECK-LABEL: func @mod_of_mod(
//       CHECK:   %[[c0:.*]] = arith.constant 0
//       CHECK:   return %[[c0]], %[[c0]]
func.func @mod_of_mod(%lb: index, %ub: index, %step: index) -> (index, index) {
  // Simplify: (ub - ub % step) % step == 0
  %0 = affine.apply affine_map<()[s0, s1] -> ((s0 - (s0 mod s1)) mod s1)> ()[%ub, %step]
  // Simplify: (ub - (ub - lb) % step - lb) % step == 0
  %1 = affine.apply affine_map<()[s0, s1, s2] -> ((s0 - ((s0 - s2) mod s1) - s2) mod s1)> ()[%ub, %step, %lb]
  return %0, %1 : index, index
}

// -----

// CHECK-LABEL:  func.func @prefetch_canonicalize
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<512xf32>) {
func.func @prefetch_canonicalize(%arg0: memref<512xf32>) -> () {
  // CHECK: affine.for [[I_0_:%.+]] = 0 to 8 {
  affine.for %arg3 = 0 to 8  {
    %1 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%arg3]
    // CHECK: affine.prefetch [[PARAM_0_]][symbol([[I_0_]]) * 64], read, locality<3>, data : memref<512xf32>
    affine.prefetch %arg0[%1], read, locality<3>, data : memref<512xf32>
  }
  return
}

// -----

func.func @drop_unit_basis_in_delinearize(%arg0 : index, %arg1 : index, %arg2 : index) ->
    (index, index, index, index, index, index) {
  %c1 = arith.constant 1 : index
  %0:6 = affine.delinearize_index %arg0 into (%c1, %arg1, %c1, %c1, %arg2, %c1)
      : index, index, index, index, index, index
  return %0#0, %0#1, %0#2, %0#3, %0#4, %0#5 : index, index, index, index, index, index
}
// CHECK-LABEL: func @drop_unit_basis_in_delinearize(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: index,
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: index,
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: index)
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[DELINEARIZE:.+]]:2 = affine.delinearize_index %[[ARG0]] into (%[[ARG1]], %[[ARG2]])
//       CHECK:   return %[[C0]], %[[DELINEARIZE]]#0, %[[C0]], %[[C0]], %[[DELINEARIZE]]#1, %[[C0]]

// -----

func.func @drop_all_unit_bases(%arg0 : index) -> (index, index) {
  %c1 = arith.constant 1 : index
  %0:2 = affine.delinearize_index %arg0 into (%c1, %c1) : index, index
  return %0#0, %0#1 : index, index
}
// CHECK-LABEL: func @drop_all_unit_bases(
//  CHECK-SAME:     %[[ARG0:.+]]: index)
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-NOT:   affine.delinearize_index
//       CHECK:   return %[[C0]], %[[C0]]

// -----

func.func @drop_single_loop_delinearize(%arg0 : index, %arg1 : index) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %2 = scf.for %iv = %c0 to %arg1 step %c1 iter_args(%arg2 = %c0) -> index {
    %0 = affine.delinearize_index %iv into (%arg1) : index
    %1 = "some_use"(%arg2, %0) : (index, index) -> (index)
    scf.yield %1 : index
  }
  return %2 : index
}
// CHECK-LABEL: func @drop_single_loop_delinearize(
//  CHECK-SAME:     %[[ARG0:.+]]: index)
//       CHECK:   scf.for %[[IV:[a-zA-Z0-9]+]] =
//   CHECK-NOT:     affine.delinearize_index
//       CHECK:     "some_use"(%{{.+}}, %[[IV]])
