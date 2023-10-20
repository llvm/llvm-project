// DEFINE: %{compile} = mlir-opt %s  -test-transform-dialect-interpreter -test-transform-dialect-erase-schedule\
// DEFINE:    -cse -canonicalize -convert-vector-to-scf -arm-sve-legalize-vector-storage\
// DEFINE:    -convert-vector-to-llvm="enable-arm-sve" -test-lower-to-llvm
// DEFINE: %{entry} =
// DEFINE: %{run} = %mcr_aarch64_cmd -e=%{entry} -entry-point-result=void --march=aarch64 --mattr="+sve" -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext

// REDEFINE: %{entry} = entry_i32
// RUN: %{compile} | %{run} | FileCheck %s --check-prefix=I32

// REDEFINE: %{entry} = entry_f32
// RUN: %{compile} | %{run} | FileCheck %s --check-prefix=F32

#matmat_accesses = [
  affine_map<(i, j, k) -> (i, k)>,
  affine_map<(i, j, k) -> (k, j)>,
  affine_map<(i, j, k) -> (i, j)>
]
#matmat_trait = {
  indexing_maps = #matmat_accesses,
  iterator_types = ["parallel", "parallel", "reduction"]
}

func.func @entry_i32() {
  %vscale = vector.vscale

  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c5 = arith.constant 5 : index
  %n_rows = arith.muli %vscale, %c2 : index

  %cst = arith.constant 0: i32
  %i32_123 = arith.constant 123 : i32
  %i32_314 = arith.constant 314 : i32

  // Allocate and initialize matrix A
  %A_alloc = memref.alloca() : memref<3x5xi32>
  linalg.fill ins(%i32_123 : i32) outs(%A_alloc :memref<3x5xi32>)
  %mask_a = vector.create_mask %c3, %c5 : vector<3x5xi1>
  %vector_a = vector.transfer_read %A_alloc[%c0, %c0], %cst, %mask_a {in_bounds = [true, true]} : memref<3x5xi32>, vector<3x5xi32>

  // Allocate and initialize matrix B
  %B_alloc = memref.alloca(%n_rows) : memref<5x?xi32>
  linalg.fill ins(%i32_123 : i32) outs(%B_alloc :memref<5x?xi32>)
  %mask_b = vector.create_mask %c5, %n_rows : vector<5x[2]xi1>
  %vector_b = vector.transfer_read %B_alloc[%c0, %c0], %cst, %mask_b {in_bounds = [true, true]} : memref<5x?xi32>, vector<5x[2]xi32>

  // Allocate and initialize matrix C
  %C_alloc = memref.alloca(%n_rows) : memref<3x?xi32>
  linalg.fill ins(%i32_314 : i32) outs(%C_alloc :memref<3x?xi32>)
  %mask_c = vector.create_mask %c3, %n_rows : vector<3x[2]xi1>
  %vector_c = vector.transfer_read %C_alloc[%c0, %c0], %cst, %mask_c {in_bounds = [true, true]} : memref<3x?xi32>, vector<3x[2]xi32>

  // Matmul
  %m = vector.create_mask %c3, %n_rows, %c5 : vector<3x[2]x5xi1>
  %0 = vector.mask %m { vector.contract #matmat_trait %vector_a, %vector_b, %vector_c
    : vector<3x5xi32>, vector<5x[2]xi32> into vector<3x[2]xi32> } : vector<3x[2]x5xi1> -> vector<3x[2]xi32>

  // Print the output
  %slice1 = vector.extract %0[0] : vector<[2]xi32> from vector<3x[2]xi32>
  // I32: ( 75959, 75959, 75959, 75959
  vector.print %slice1 : vector<[2]xi32>
  %slice2 = vector.extract %0[1] : vector<[2]xi32> from vector<3x[2]xi32>
  // I32-NEXT: ( 75959, 75959, 75959, 75959
  vector.print %slice2 : vector<[2]xi32>
  %slice3 = vector.extract %0[2] : vector<[2]xi32> from vector<3x[2]xi32>
  // I32-NEXT: ( 75959, 75959, 75959, 75959
  vector.print %slice3 : vector<[2]xi32>

  // CHECK: SVE: END OF TEST OUTPUT
  vector.print str "SVE: END OF TEST OUTPUT"

  return
}

func.func @entry_f32() {
  %vscale = vector.vscale

  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c5 = arith.constant 5 : index
  %n_rows = arith.muli %vscale, %c2 : index

  %cst = arith.constant 0.0: f32
  %f32_123 = arith.constant 1.23 : f32
  %f32_314 = arith.constant 3.14 : f32

  // Allocate and initialize matrix A
  %A_alloc = memref.alloca() : memref<3x5xf32>
  linalg.fill ins(%f32_123 : f32) outs(%A_alloc :memref<3x5xf32>)
  %mask_a = vector.create_mask %c3, %c5 : vector<3x5xi1>
  %vector_a = vector.transfer_read %A_alloc[%c0, %c0], %cst, %mask_a {in_bounds = [true, true]} : memref<3x5xf32>, vector<3x5xf32>

  // Allocate and initialize matrix B
  %B_alloc = memref.alloca(%n_rows) : memref<5x?xf32>
  linalg.fill ins(%f32_123 : f32) outs(%B_alloc :memref<5x?xf32>)
  %mask_b = vector.create_mask %c5, %n_rows : vector<5x[2]xi1>
  %vector_b = vector.transfer_read %B_alloc[%c0, %c0], %cst, %mask_b {in_bounds = [true, true]} : memref<5x?xf32>, vector<5x[2]xf32>

  // Allocate and initialize matrix C
  %C_alloc = memref.alloca(%n_rows) : memref<3x?xf32>
  linalg.fill ins(%f32_314 : f32) outs(%C_alloc :memref<3x?xf32>)
  %mask_c = vector.create_mask %c3, %n_rows : vector<3x[2]xi1>
  %vector_c = vector.transfer_read %C_alloc[%c0, %c0], %cst, %mask_c {in_bounds = [true, true]} : memref<3x?xf32>, vector<3x[2]xf32>

  // Matmul
  %m = vector.create_mask %c3, %n_rows, %c5 : vector<3x[2]x5xi1>
  %0 = vector.mask %m { vector.contract #matmat_trait %vector_a, %vector_b, %vector_c
    : vector<3x5xf32>, vector<5x[2]xf32> into vector<3x[2]xf32> } : vector<3x[2]x5xi1> -> vector<3x[2]xf32>

  // Print the output
  %slice1 = vector.extract %0[0] : vector<[2]xf32> from vector<3x[2]xf32>
  // F32: ( 10.7045, 10.7045, 10.7045, 10.7045
  vector.print %slice1 : vector<[2]xf32>
  %slice2 = vector.extract %0[1] : vector<[2]xf32> from vector<3x[2]xf32>
  // F32-NEXT: ( 10.7045, 10.7045, 10.7045, 10.7045
  vector.print %slice2 : vector<[2]xf32>
  %slice3 = vector.extract %0[2] : vector<[2]xf32> from vector<3x[2]xf32>
  // F32-NEXT: ( 10.7045, 10.7045, 10.7045, 10.7045
  vector.print %slice3 : vector<[2]xf32>

  // CHECK: SVE: END OF TEST OUTPUT
  vector.print str "SVE: END OF TEST OUTPUT"

  return
}

transform.sequence failures(propagate) {
^bb1(%module_op: !transform.any_op):
  %f = transform.structured.match ops{["func.func"]} in %module_op
    : (!transform.any_op) -> !transform.any_op

  transform.apply_patterns to %f {
    transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"
  } : !transform.any_op
}
