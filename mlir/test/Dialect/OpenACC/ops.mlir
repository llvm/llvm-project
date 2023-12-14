// RUN: mlir-opt -split-input-file %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: mlir-opt -split-input-file %s | mlir-opt  -split-input-file  | FileCheck %s
// Verify the generic form can be parsed.
// RUN: mlir-opt -split-input-file -mlir-print-op-generic %s | mlir-opt -split-input-file | FileCheck %s

func.func @compute1(%A: memref<10x10xf32>, %B: memref<10x10xf32>, %C: memref<10x10xf32>) -> memref<10x10xf32> {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  %async = arith.constant 1 : i64

  acc.parallel async(%async: i64) {
    acc.loop gang vector {
      scf.for %arg3 = %c0 to %c10 step %c1 {
        scf.for %arg4 = %c0 to %c10 step %c1 {
          scf.for %arg5 = %c0 to %c10 step %c1 {
            %a = memref.load %A[%arg3, %arg5] : memref<10x10xf32>
            %b = memref.load %B[%arg5, %arg4] : memref<10x10xf32>
            %cij = memref.load %C[%arg3, %arg4] : memref<10x10xf32>
            %p = arith.mulf %a, %b : f32
            %co = arith.addf %cij, %p : f32
            memref.store %co, %C[%arg3, %arg4] : memref<10x10xf32>
          }
        }
      }
      acc.yield
    } attributes { collapse = 3 }
    acc.yield
  }

  return %C : memref<10x10xf32>
}

// CHECK-LABEL: func @compute1(
//  CHECK-NEXT:   %{{.*}} = arith.constant 0 : index
//  CHECK-NEXT:   %{{.*}} = arith.constant 10 : index
//  CHECK-NEXT:   %{{.*}} = arith.constant 1 : index
//  CHECK-NEXT:   [[ASYNC:%.*]] = arith.constant 1 : i64
//  CHECK-NEXT:   acc.parallel async([[ASYNC]] : i64) {
//  CHECK-NEXT:     acc.loop gang vector {
//  CHECK-NEXT:       scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//  CHECK-NEXT:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//  CHECK-NEXT:           scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//  CHECK-NEXT:             %{{.*}} = memref.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<10x10xf32>
//  CHECK-NEXT:             %{{.*}} = memref.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<10x10xf32>
//  CHECK-NEXT:             %{{.*}} = memref.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<10x10xf32>
//  CHECK-NEXT:             %{{.*}} = arith.mulf %{{.*}}, %{{.*}} : f32
//  CHECK-NEXT:             %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f32
//  CHECK-NEXT:             memref.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<10x10xf32>
//  CHECK-NEXT:           }
//  CHECK-NEXT:         }
//  CHECK-NEXT:       }
//  CHECK-NEXT:       acc.yield
//  CHECK-NEXT:     } attributes {collapse = 3 : i64}
//  CHECK-NEXT:     acc.yield
//  CHECK-NEXT:   }
//  CHECK-NEXT:   return %{{.*}} : memref<10x10xf32>
//  CHECK-NEXT: }

// -----

func.func @compute2(%A: memref<10x10xf32>, %B: memref<10x10xf32>, %C: memref<10x10xf32>) -> memref<10x10xf32> {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index

  acc.parallel {
    acc.loop {
      scf.for %arg3 = %c0 to %c10 step %c1 {
        scf.for %arg4 = %c0 to %c10 step %c1 {
          scf.for %arg5 = %c0 to %c10 step %c1 {
            %a = memref.load %A[%arg3, %arg5] : memref<10x10xf32>
            %b = memref.load %B[%arg5, %arg4] : memref<10x10xf32>
            %cij = memref.load %C[%arg3, %arg4] : memref<10x10xf32>
            %p = arith.mulf %a, %b : f32
            %co = arith.addf %cij, %p : f32
            memref.store %co, %C[%arg3, %arg4] : memref<10x10xf32>
          }
        }
      }
      acc.yield
    } attributes {seq}
    acc.yield
  }

  return %C : memref<10x10xf32>
}

// CHECK-LABEL: func @compute2(
//  CHECK-NEXT:   %{{.*}} = arith.constant 0 : index
//  CHECK-NEXT:   %{{.*}} = arith.constant 10 : index
//  CHECK-NEXT:   %{{.*}} = arith.constant 1 : index
//  CHECK-NEXT:   acc.parallel {
//  CHECK-NEXT:     acc.loop {
//  CHECK-NEXT:       scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//  CHECK-NEXT:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//  CHECK-NEXT:           scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//  CHECK-NEXT:             %{{.*}} = memref.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<10x10xf32>
//  CHECK-NEXT:             %{{.*}} = memref.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<10x10xf32>
//  CHECK-NEXT:             %{{.*}} = memref.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<10x10xf32>
//  CHECK-NEXT:             %{{.*}} = arith.mulf %{{.*}}, %{{.*}} : f32
//  CHECK-NEXT:             %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f32
//  CHECK-NEXT:             memref.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<10x10xf32>
//  CHECK-NEXT:           }
//  CHECK-NEXT:         }
//  CHECK-NEXT:       }
//  CHECK-NEXT:       acc.yield
//  CHECK-NEXT:     } attributes {seq}
//  CHECK-NEXT:     acc.yield
//  CHECK-NEXT:   }
//  CHECK-NEXT:   return %{{.*}} : memref<10x10xf32>
//  CHECK-NEXT: }

// -----

acc.private.recipe @privatization_memref_10_f32 : memref<10xf32> init {
^bb0(%arg0: memref<10xf32>):
  %0 = memref.alloc() : memref<10xf32>
  acc.yield %0 : memref<10xf32>
} destroy {
^bb0(%arg0: memref<10xf32>):
  memref.dealloc %arg0 : memref<10xf32> 
  acc.terminator
}

func.func @compute3(%a: memref<10x10xf32>, %b: memref<10x10xf32>, %c: memref<10xf32>, %d: memref<10xf32>) -> memref<10xf32> {
  %lb = arith.constant 0 : index
  %st = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %numGangs = arith.constant 10 : i64
  %numWorkers = arith.constant 10 : i64

  %pa = acc.present varPtr(%a : memref<10x10xf32>) -> memref<10x10xf32>
  %pb = acc.present varPtr(%b : memref<10x10xf32>) -> memref<10x10xf32>
  %pc = acc.present varPtr(%c : memref<10xf32>) -> memref<10xf32>
  %pd = acc.present varPtr(%d : memref<10xf32>) -> memref<10xf32>
  acc.data dataOperands(%pa, %pb, %pc, %pd: memref<10x10xf32>, memref<10x10xf32>, memref<10xf32>, memref<10xf32>) {
    %private = acc.private varPtr(%c : memref<10xf32>) -> memref<10xf32>
    acc.parallel num_gangs(%numGangs: i64) num_workers(%numWorkers: i64) private(@privatization_memref_10_f32 -> %private : memref<10xf32>) {
      acc.loop gang {
        scf.for %x = %lb to %c10 step %st {
          acc.loop worker {
            scf.for %y = %lb to %c10 step %st {
              %axy = memref.load %a[%x, %y] : memref<10x10xf32>
              %bxy = memref.load %b[%x, %y] : memref<10x10xf32>
              %tmp = arith.addf %axy, %bxy : f32
              memref.store %tmp, %c[%y] : memref<10xf32>
            }
            acc.yield
          }

          acc.loop {
            // for i = 0 to 10 step 1
            //   d[x] += c[i]
            scf.for %i = %lb to %c10 step %st {
              %ci = memref.load %c[%i] : memref<10xf32>
              %dx = memref.load %d[%x] : memref<10xf32>
              %z = arith.addf %ci, %dx : f32
              memref.store %z, %d[%x] : memref<10xf32>
            }
            acc.yield
          } attributes {seq}
        }
        acc.yield
      }
      acc.yield
    }
    acc.terminator
  }

  return %d : memref<10xf32>
}

// CHECK:      func @compute3({{.*}}: memref<10x10xf32>, {{.*}}: memref<10x10xf32>, [[ARG2:%.*]]: memref<10xf32>, {{.*}}: memref<10xf32>) -> memref<10xf32> {
// CHECK-NEXT:   [[C0:%.*]] = arith.constant 0 : index
// CHECK-NEXT:   [[C1:%.*]] = arith.constant 1 : index
// CHECK-NEXT:   [[C10:%.*]] = arith.constant 10 : index
// CHECK-NEXT:   [[NUMGANG:%.*]] = arith.constant 10 : i64
// CHECK-NEXT:   [[NUMWORKERS:%.*]] = arith.constant 10 : i64
// CHECK:        acc.data dataOperands(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : memref<10x10xf32>, memref<10x10xf32>, memref<10xf32>, memref<10xf32>) {
// CHECK-NEXT:     %[[P_ARG2:.*]] = acc.private varPtr([[ARG2]] : memref<10xf32>) -> memref<10xf32> 
// CHECK-NEXT:     acc.parallel num_gangs([[NUMGANG]] : i64) num_workers([[NUMWORKERS]] : i64) private(@privatization_memref_10_f32 -> %[[P_ARG2]] : memref<10xf32>) {
// CHECK-NEXT:       acc.loop gang {
// CHECK-NEXT:         scf.for %{{.*}} = [[C0]] to [[C10]] step [[C1]] {
// CHECK-NEXT:           acc.loop worker {
// CHECK-NEXT:             scf.for %{{.*}} = [[C0]] to [[C10]] step [[C1]] {
// CHECK-NEXT:               %{{.*}} = memref.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<10x10xf32>
// CHECK-NEXT:               %{{.*}} = memref.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<10x10xf32>
// CHECK-NEXT:               %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:               memref.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
// CHECK-NEXT:             }
// CHECK-NEXT:             acc.yield
// CHECK-NEXT:           }
// CHECK-NEXT:           acc.loop {
// CHECK-NEXT:             scf.for %{{.*}} = [[C0]] to [[C10]] step [[C1]] {
// CHECK-NEXT:               %{{.*}} = memref.load %{{.*}}[%{{.*}}] : memref<10xf32>
// CHECK-NEXT:               %{{.*}} = memref.load %{{.*}}[%{{.*}}] : memref<10xf32>
// CHECK-NEXT:               %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:               memref.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
// CHECK-NEXT:             }
// CHECK-NEXT:             acc.yield
// CHECK-NEXT:           } attributes {seq}
// CHECK-NEXT:         }
// CHECK-NEXT:         acc.yield
// CHECK-NEXT:       }
// CHECK-NEXT:       acc.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     acc.terminator
// CHECK-NEXT:   }
// CHECK-NEXT:   return %{{.*}} : memref<10xf32>
// CHECK-NEXT: }

// -----

func.func @testloopop(%a : memref<10xf32>) -> () {
  %i64Value = arith.constant 1 : i64
  %i32Value = arith.constant 128 : i32
  %idxValue = arith.constant 8 : index

  acc.loop gang worker vector {
    "test.openacc_dummy_op"() : () -> ()
    acc.yield
  }
  acc.loop gang(num=%i64Value: i64) {
    "test.openacc_dummy_op"() : () -> ()
    acc.yield
  }
  acc.loop gang(static=%i64Value: i64) {
    "test.openacc_dummy_op"() : () -> ()
    acc.yield
  }
  acc.loop worker(%i64Value: i64) {
    "test.openacc_dummy_op"() : () -> ()
    acc.yield
  }
  acc.loop worker(%i32Value: i32) {
    "test.openacc_dummy_op"() : () -> ()
    acc.yield
  }
  acc.loop worker(%idxValue: index) {
    "test.openacc_dummy_op"() : () -> ()
    acc.yield
  }
  acc.loop vector(%i64Value: i64) {
    "test.openacc_dummy_op"() : () -> ()
    acc.yield
  }
  acc.loop vector(%i32Value: i32) {
    "test.openacc_dummy_op"() : () -> ()
    acc.yield
  }
  acc.loop vector(%idxValue: index) {
    "test.openacc_dummy_op"() : () -> ()
    acc.yield
  }
  acc.loop gang(num=%i64Value: i64) worker vector {
    "test.openacc_dummy_op"() : () -> ()
    acc.yield
  }
  acc.loop gang(num=%i64Value: i64, static=%i64Value: i64) worker(%i64Value: i64) vector(%i64Value: i64) {
    "test.openacc_dummy_op"() : () -> ()
    acc.yield
  }
  acc.loop gang(num=%i32Value: i32, static=%idxValue: index) {
    "test.openacc_dummy_op"() : () -> ()
    acc.yield
  }
  acc.loop tile(%i64Value, %i64Value : i64, i64) {
    "test.openacc_dummy_op"() : () -> ()
    acc.yield
  }
  acc.loop tile(%i32Value, %i32Value : i32, i32) {
    "test.openacc_dummy_op"() : () -> ()
    acc.yield
  }
  acc.loop gang(static=%i64Value: i64, num=%i64Value: i64) {
    "test.openacc_dummy_op"() : () -> ()
    acc.yield
  }
  acc.loop gang(dim=%i64Value : i64, static=%i64Value: i64) {
    "test.openacc_dummy_op"() : () -> ()
    acc.yield
  }
  %b = acc.cache varPtr(%a : memref<10xf32>) -> memref<10xf32>
  acc.loop cache(%b : memref<10xf32>) {
    "test.openacc_dummy_op"() : () -> ()
    acc.yield
  }
  return
}

// CHECK:      [[I64VALUE:%.*]] = arith.constant 1 : i64
// CHECK-NEXT: [[I32VALUE:%.*]] = arith.constant 128 : i32
// CHECK-NEXT: [[IDXVALUE:%.*]] = arith.constant 8 : index
// CHECK:      acc.loop gang worker vector {
// CHECK-NEXT:   "test.openacc_dummy_op"() : () -> ()
// CHECK-NEXT:   acc.yield
// CHECK-NEXT: }
// CHECK:      acc.loop gang(num=[[I64VALUE]] : i64) {
// CHECK-NEXT:   "test.openacc_dummy_op"() : () -> ()
// CHECK-NEXT:   acc.yield
// CHECK-NEXT: }
// CHECK:      acc.loop gang(static=[[I64VALUE]] : i64) {
// CHECK-NEXT:   "test.openacc_dummy_op"() : () -> ()
// CHECK-NEXT:   acc.yield
// CHECK-NEXT: }
// CHECK:      acc.loop worker([[I64VALUE]] : i64) {
// CHECK-NEXT:   "test.openacc_dummy_op"() : () -> ()
// CHECK-NEXT:   acc.yield
// CHECK-NEXT: }
// CHECK:      acc.loop worker([[I32VALUE]] : i32) {
// CHECK-NEXT:   "test.openacc_dummy_op"() : () -> ()
// CHECK-NEXT:   acc.yield
// CHECK-NEXT: }
// CHECK:      acc.loop worker([[IDXVALUE]] : index) {
// CHECK-NEXT:   "test.openacc_dummy_op"() : () -> ()
// CHECK-NEXT:   acc.yield
// CHECK-NEXT: }
// CHECK:      acc.loop vector([[I64VALUE]] : i64) {
// CHECK-NEXT:   "test.openacc_dummy_op"() : () -> ()
// CHECK-NEXT:   acc.yield
// CHECK-NEXT: }
// CHECK:      acc.loop vector([[I32VALUE]] : i32) {
// CHECK-NEXT:   "test.openacc_dummy_op"() : () -> ()
// CHECK-NEXT:   acc.yield
// CHECK-NEXT: }
// CHECK:      acc.loop vector([[IDXVALUE]] : index) {
// CHECK-NEXT:   "test.openacc_dummy_op"() : () -> ()
// CHECK-NEXT:   acc.yield
// CHECK-NEXT: }
// CHECK:      acc.loop gang(num=[[I64VALUE]] : i64) worker vector {
// CHECK-NEXT:   "test.openacc_dummy_op"() : () -> ()
// CHECK-NEXT:   acc.yield
// CHECK-NEXT: }
// CHECK:      acc.loop gang(num=[[I64VALUE]] : i64, static=[[I64VALUE]] : i64) worker([[I64VALUE]] : i64) vector([[I64VALUE]] : i64) {
// CHECK-NEXT:   "test.openacc_dummy_op"() : () -> ()
// CHECK-NEXT:   acc.yield
// CHECK-NEXT: }
// CHECK:      acc.loop gang(num=[[I32VALUE]] : i32, static=[[IDXVALUE]] : index) {
// CHECK-NEXT:   "test.openacc_dummy_op"() : () -> ()
// CHECK-NEXT:   acc.yield
// CHECK-NEXT: }
// CHECK:      acc.loop tile([[I64VALUE]], [[I64VALUE]] : i64, i64) {
// CHECK-NEXT:   "test.openacc_dummy_op"() : () -> ()
// CHECK-NEXT:   acc.yield
// CHECK-NEXT: }
// CHECK:      acc.loop tile([[I32VALUE]], [[I32VALUE]] : i32, i32) {
// CHECK-NEXT:   "test.openacc_dummy_op"() : () -> ()
// CHECK-NEXT:   acc.yield
// CHECK-NEXT: }
// CHECK:      acc.loop gang(num=[[I64VALUE]] : i64, static=[[I64VALUE]] : i64) {
// CHECK-NEXT:   "test.openacc_dummy_op"() : () -> ()
// CHECK-NEXT:   acc.yield
// CHECK-NEXT: }
// CHECK:      acc.loop gang(dim=[[I64VALUE]] : i64, static=[[I64VALUE]] : i64) {
// CHECK-NEXT:   "test.openacc_dummy_op"() : () -> ()
// CHECK-NEXT:   acc.yield
// CHECK-NEXT: }
// CHECK:      %{{.*}} = acc.cache varPtr(%{{.*}} : memref<10xf32>) -> memref<10xf32>
// CHECK-NEXT: acc.loop cache(%{{.*}} : memref<10xf32>) {
// CHECK-NEXT:   "test.openacc_dummy_op"() : () -> ()
// CHECK-NEXT:   acc.yield
// CHECK-NEXT: }

// -----

func.func @acc_loop_multiple_block() {
  acc.parallel {
    acc.loop {
      %c1 = arith.constant 1 : index
      cf.br ^bb1(%c1 : index)
    ^bb1(%9: index):
      %c0 = arith.constant 0 : index
      %12 = arith.cmpi sgt, %9, %c0 : index
      cf.cond_br %12, ^bb2, ^bb3
    ^bb2:
      %c1_0 = arith.constant 1 : index
      %c10 = arith.constant 10 : index
      %22 = arith.subi %c10, %c1_0 : index
      cf.br ^bb1(%22 : index)
    ^bb3:
      acc.yield
    }
    acc.yield
  }
  return
}

// CHECK-LABEL: func.func @acc_loop_multiple_block()
// CHECK: acc.parallel
// CHECK: acc.loop
// CHECK-3: ^bb{{.*}}
// CHECK: acc.yield
// CHECK: acc.yield

// -----

acc.private.recipe @privatization_memref_10_f32 : memref<10xf32> init {
^bb0(%arg0: memref<10xf32>):
  %0 = memref.alloc() : memref<10xf32>
  acc.yield %0 : memref<10xf32>
} destroy {
^bb0(%arg0: memref<10xf32>):
  memref.dealloc %arg0 : memref<10xf32> 
  acc.terminator
}

acc.private.recipe @privatization_memref_10_10_f32 : memref<10x10xf32> init {
^bb0(%arg0: memref<10x10xf32>):
  %0 = memref.alloc() : memref<10x10xf32>
  acc.yield %0 : memref<10x10xf32>
} destroy {
^bb0(%arg0: memref<10x10xf32>):
  memref.dealloc %arg0 : memref<10x10xf32> 
  acc.terminator
}

acc.firstprivate.recipe @privatization_memref_10xf32 : memref<10xf32> init {
^bb0(%arg0: memref<10xf32>):
  %0 = memref.alloc() : memref<10xf32>
  acc.yield %0 : memref<10xf32>
} copy {
^bb0(%arg0: memref<10xf32>, %arg1: memref<10xf32>):
  acc.terminator
} destroy {
^bb0(%arg0: memref<10xf32>):
  memref.dealloc %arg0 : memref<10xf32> 
  acc.terminator
}

func.func @testparallelop(%a: memref<10xf32>, %b: memref<10xf32>, %c: memref<10x10xf32>) -> () {
  %i64value = arith.constant 1 : i64
  %i32value = arith.constant 1 : i32
  %idxValue = arith.constant 1 : index
  acc.parallel async(%i64value: i64) {
  }
  acc.parallel async(%i32value: i32) {
  }
  acc.parallel async(%idxValue: index) {
  }
  acc.parallel wait(%i64value: i64) {
  }
  acc.parallel wait(%i32value: i32) {
  }
  acc.parallel wait(%idxValue: index) {
  }
  acc.parallel wait(%i64value, %i32value, %idxValue : i64, i32, index) {
  }
  acc.parallel num_gangs(%i64value: i64) {
  }
  acc.parallel num_gangs(%i32value: i32) {
  }
  acc.parallel num_gangs(%idxValue: index) {
  }
  acc.parallel num_gangs(%i64value, %i64value, %idxValue : i64, i64, index) {
  }
  acc.parallel num_workers(%i64value: i64) {
  }
  acc.parallel num_workers(%i32value: i32) {
  }
  acc.parallel num_workers(%idxValue: index) {
  }
  acc.parallel vector_length(%i64value: i64) {
  }
  acc.parallel vector_length(%i32value: i32) {
  }
  acc.parallel vector_length(%idxValue: index) {
  }
  acc.parallel private(@privatization_memref_10_f32 -> %a : memref<10xf32>, @privatization_memref_10_10_f32 -> %c : memref<10x10xf32>) firstprivate(@privatization_memref_10xf32 -> %b: memref<10xf32>) {
  }
  acc.parallel {
  } attributes {defaultAttr = #acc<defaultvalue none>}
  acc.parallel {
  } attributes {defaultAttr = #acc<defaultvalue present>}
  acc.parallel {
  } attributes {asyncAttr}
  acc.parallel {
  } attributes {waitAttr}
  acc.parallel {
  } attributes {selfAttr}
  return
}

// CHECK:      func @testparallelop([[ARGA:%.*]]: memref<10xf32>, [[ARGB:%.*]]: memref<10xf32>, [[ARGC:%.*]]: memref<10x10xf32>) {
// CHECK:      [[I64VALUE:%.*]] = arith.constant 1 : i64
// CHECK:      [[I32VALUE:%.*]] = arith.constant 1 : i32
// CHECK:      [[IDXVALUE:%.*]] = arith.constant 1 : index
// CHECK:      acc.parallel async([[I64VALUE]] : i64) {
// CHECK-NEXT: }
// CHECK:      acc.parallel async([[I32VALUE]] : i32) {
// CHECK-NEXT: }
// CHECK:      acc.parallel async([[IDXVALUE]] : index) {
// CHECK-NEXT: }
// CHECK:      acc.parallel wait([[I64VALUE]] : i64) {
// CHECK-NEXT: }
// CHECK:      acc.parallel wait([[I32VALUE]] : i32) {
// CHECK-NEXT: }
// CHECK:      acc.parallel wait([[IDXVALUE]] : index) {
// CHECK-NEXT: }
// CHECK:      acc.parallel wait([[I64VALUE]], [[I32VALUE]], [[IDXVALUE]] : i64, i32, index) {
// CHECK-NEXT: }
// CHECK:      acc.parallel num_gangs([[I64VALUE]] : i64) {
// CHECK-NEXT: }
// CHECK:      acc.parallel num_gangs([[I32VALUE]] : i32) {
// CHECK-NEXT: }
// CHECK:      acc.parallel num_gangs([[IDXVALUE]] : index) {
// CHECK-NEXT: }
// CHECK:      acc.parallel num_gangs([[I64VALUE]], [[I64VALUE]], [[IDXVALUE]] : i64, i64, index) {
// CHECK-NEXT: }
// CHECK:      acc.parallel num_workers([[I64VALUE]] : i64) {
// CHECK-NEXT: }
// CHECK:      acc.parallel num_workers([[I32VALUE]] : i32) {
// CHECK-NEXT: }
// CHECK:      acc.parallel num_workers([[IDXVALUE]] : index) {
// CHECK-NEXT: }
// CHECK:      acc.parallel vector_length([[I64VALUE]] : i64) {
// CHECK-NEXT: }
// CHECK:      acc.parallel vector_length([[I32VALUE]] : i32) {
// CHECK-NEXT: }
// CHECK:      acc.parallel vector_length([[IDXVALUE]] : index) {
// CHECK-NEXT: }
// CHECK:      acc.parallel firstprivate(@privatization_memref_10xf32 -> [[ARGB]] : memref<10xf32>) private(@privatization_memref_10_f32 -> [[ARGA]] : memref<10xf32>, @privatization_memref_10_10_f32 -> [[ARGC]] : memref<10x10xf32>) {
// CHECK-NEXT: }
// CHECK:      acc.parallel {
// CHECK-NEXT: } attributes {defaultAttr = #acc<defaultvalue none>}
// CHECK:      acc.parallel {
// CHECK-NEXT: } attributes {defaultAttr = #acc<defaultvalue present>}
// CHECK:      acc.parallel {
// CHECK-NEXT: } attributes {asyncAttr}
// CHECK:      acc.parallel {
// CHECK-NEXT: } attributes {waitAttr}
// CHECK:      acc.parallel {
// CHECK-NEXT: } attributes {selfAttr}

// -----

acc.private.recipe @privatization_memref_10_f32 : memref<10xf32> init {
^bb0(%arg0: memref<10xf32>):
  %0 = memref.alloc() : memref<10xf32>
  acc.yield %0 : memref<10xf32>
} destroy {
^bb0(%arg0: memref<10xf32>):
  memref.dealloc %arg0 : memref<10xf32> 
  acc.terminator
}

acc.private.recipe @privatization_memref_10_10_f32 : memref<10x10xf32> init {
^bb0(%arg0: memref<10x10xf32>):
  %0 = memref.alloc() : memref<10x10xf32>
  acc.yield %0 : memref<10x10xf32>
} destroy {
^bb0(%arg0: memref<10x10xf32>):
  memref.dealloc %arg0 : memref<10x10xf32> 
  acc.terminator
}

// Test optional destroy region
acc.firstprivate.recipe @firstprivatization_memref_20xf32 : memref<20xf32> init {
^bb0(%arg0: memref<20xf32>):
  %0 = memref.alloc() : memref<20xf32>
  acc.yield %0 : memref<20xf32>
} copy {
^bb0(%arg0: memref<20xf32>, %arg1: memref<20xf32>):
  acc.terminator
}

// CHECK-LABEL: acc.firstprivate.recipe @firstprivatization_memref_20xf32 : memref<20xf32> init

acc.firstprivate.recipe @firstprivatization_memref_10xf32 : memref<10xf32> init {
^bb0(%arg0: memref<10xf32>):
  %0 = memref.alloc() : memref<10xf32>
  acc.yield %0 : memref<10xf32>
} copy {
^bb0(%arg0: memref<10xf32>, %arg1: memref<10xf32>):
  acc.terminator
} destroy {
^bb0(%arg0: memref<10xf32>):
  memref.dealloc %arg0 : memref<10xf32> 
  acc.terminator
}

func.func @testserialop(%a: memref<10xf32>, %b: memref<10xf32>, %c: memref<10x10xf32>) -> () {
  %i64value = arith.constant 1 : i64
  %i32value = arith.constant 1 : i32
  %idxValue = arith.constant 1 : index
  acc.serial async(%i64value: i64) {
  }
  acc.serial async(%i32value: i32) {
  }
  acc.serial async(%idxValue: index) {
  }
  acc.serial wait(%i64value: i64) {
  }
  acc.serial wait(%i32value: i32) {
  }
  acc.serial wait(%idxValue: index) {
  }
  acc.serial wait(%i64value, %i32value, %idxValue : i64, i32, index) {
  }
  %firstprivate = acc.firstprivate varPtr(%b : memref<10xf32>) -> memref<10xf32>
  acc.serial private(@privatization_memref_10_f32 -> %a : memref<10xf32>, @privatization_memref_10_10_f32 -> %c : memref<10x10xf32>) firstprivate(@firstprivatization_memref_10xf32 -> %firstprivate : memref<10xf32>) {
  }
  acc.serial {
  } attributes {defaultAttr = #acc<defaultvalue none>}
  acc.serial {
  } attributes {defaultAttr = #acc<defaultvalue present>}
  acc.serial {
  } attributes {asyncAttr}
  acc.serial {
  } attributes {waitAttr}
  acc.serial {
  } attributes {selfAttr}
  acc.serial {
    acc.yield
  } attributes {selfAttr}
  return
}

// CHECK:      func @testserialop([[ARGA:%.*]]: memref<10xf32>, [[ARGB:%.*]]: memref<10xf32>, [[ARGC:%.*]]: memref<10x10xf32>) {
// CHECK:      [[I64VALUE:%.*]] = arith.constant 1 : i64
// CHECK:      [[I32VALUE:%.*]] = arith.constant 1 : i32
// CHECK:      [[IDXVALUE:%.*]] = arith.constant 1 : index
// CHECK:      acc.serial async([[I64VALUE]] : i64) {
// CHECK-NEXT: }
// CHECK:      acc.serial async([[I32VALUE]] : i32) {
// CHECK-NEXT: }
// CHECK:      acc.serial async([[IDXVALUE]] : index) {
// CHECK-NEXT: }
// CHECK:      acc.serial wait([[I64VALUE]] : i64) {
// CHECK-NEXT: }
// CHECK:      acc.serial wait([[I32VALUE]] : i32) {
// CHECK-NEXT: }
// CHECK:      acc.serial wait([[IDXVALUE]] : index) {
// CHECK-NEXT: }
// CHECK:      acc.serial wait([[I64VALUE]], [[I32VALUE]], [[IDXVALUE]] : i64, i32, index) {
// CHECK-NEXT: }
// CHECK:      %[[FIRSTP:.*]] = acc.firstprivate varPtr([[ARGB]] : memref<10xf32>) -> memref<10xf32>
// CHECK:      acc.serial firstprivate(@firstprivatization_memref_10xf32 -> %[[FIRSTP]] : memref<10xf32>) private(@privatization_memref_10_f32 -> [[ARGA]] : memref<10xf32>, @privatization_memref_10_10_f32 -> [[ARGC]] : memref<10x10xf32>) {
// CHECK-NEXT: }
// CHECK:      acc.serial {
// CHECK-NEXT: } attributes {defaultAttr = #acc<defaultvalue none>}
// CHECK:      acc.serial {
// CHECK-NEXT: } attributes {defaultAttr = #acc<defaultvalue present>}
// CHECK:      acc.serial {
// CHECK-NEXT: } attributes {asyncAttr}
// CHECK:      acc.serial {
// CHECK-NEXT: } attributes {waitAttr}
// CHECK:      acc.serial {
// CHECK-NEXT: } attributes {selfAttr}
// CHECK:      acc.serial {
// CHECK:        acc.yield
// CHECK-NEXT: } attributes {selfAttr}

// -----


func.func @testserialop(%a: memref<10xf32>, %b: memref<10xf32>, %c: memref<10x10xf32>) -> () {
  %i64value = arith.constant 1 : i64
  %i32value = arith.constant 1 : i32
  %idxValue = arith.constant 1 : index
  acc.kernels async(%i64value: i64) {
  }
  acc.kernels async(%i32value: i32) {
  }
  acc.kernels async(%idxValue: index) {
  }
  acc.kernels wait(%i64value: i64) {
  }
  acc.kernels wait(%i32value: i32) {
  }
  acc.kernels wait(%idxValue: index) {
  }
  acc.kernels wait(%i64value, %i32value, %idxValue : i64, i32, index) {
  }
  acc.kernels {
  } attributes {defaultAttr = #acc<defaultvalue none>}
  acc.kernels {
  } attributes {defaultAttr = #acc<defaultvalue present>}
  acc.kernels {
  } attributes {asyncAttr}
  acc.kernels {
  } attributes {waitAttr}
  acc.kernels {
  } attributes {selfAttr}
  acc.kernels {
    acc.terminator
  } attributes {selfAttr}
  return
}

// CHECK:      func @testserialop([[ARGA:%.*]]: memref<10xf32>, [[ARGB:%.*]]: memref<10xf32>, [[ARGC:%.*]]: memref<10x10xf32>) {
// CHECK:      [[I64VALUE:%.*]] = arith.constant 1 : i64
// CHECK:      [[I32VALUE:%.*]] = arith.constant 1 : i32
// CHECK:      [[IDXVALUE:%.*]] = arith.constant 1 : index
// CHECK:      acc.kernels async([[I64VALUE]] : i64) {
// CHECK-NEXT: }
// CHECK:      acc.kernels async([[I32VALUE]] : i32) {
// CHECK-NEXT: }
// CHECK:      acc.kernels async([[IDXVALUE]] : index) {
// CHECK-NEXT: }
// CHECK:      acc.kernels wait([[I64VALUE]] : i64) {
// CHECK-NEXT: }
// CHECK:      acc.kernels wait([[I32VALUE]] : i32) {
// CHECK-NEXT: }
// CHECK:      acc.kernels wait([[IDXVALUE]] : index) {
// CHECK-NEXT: }
// CHECK:      acc.kernels wait([[I64VALUE]], [[I32VALUE]], [[IDXVALUE]] : i64, i32, index) {
// CHECK-NEXT: }
// CHECK:      acc.kernels {
// CHECK-NEXT: } attributes {defaultAttr = #acc<defaultvalue none>}
// CHECK:      acc.kernels {
// CHECK-NEXT: } attributes {defaultAttr = #acc<defaultvalue present>}
// CHECK:      acc.kernels {
// CHECK-NEXT: } attributes {asyncAttr}
// CHECK:      acc.kernels {
// CHECK-NEXT: } attributes {waitAttr}
// CHECK:      acc.kernels {
// CHECK-NEXT: } attributes {selfAttr}
// CHECK:      acc.kernels {
// CHECK:        acc.terminator
// CHECK-NEXT: } attributes {selfAttr}

// -----

func.func @testdataop(%a: memref<f32>, %b: memref<f32>, %c: memref<f32>) -> () {
  %ifCond = arith.constant true

  %0 = acc.present varPtr(%a : memref<f32>) -> memref<f32>
  acc.data if(%ifCond) dataOperands(%0 : memref<f32>) {
  }

  %1 = acc.present varPtr(%a : memref<f32>) -> memref<f32>
  acc.data dataOperands(%1 : memref<f32>) if(%ifCond) {
  }

  %2 = acc.present varPtr(%a : memref<f32>) -> memref<f32>
  %3 = acc.present varPtr(%b : memref<f32>) -> memref<f32>
  %4 = acc.present varPtr(%c : memref<f32>) -> memref<f32>
  acc.data dataOperands(%2, %3, %4 : memref<f32>, memref<f32>, memref<f32>) {
  }

  %5 = acc.copyin varPtr(%a : memref<f32>) -> memref<f32>
  %6 = acc.copyin varPtr(%b : memref<f32>) -> memref<f32>
  %7 = acc.copyin varPtr(%c : memref<f32>) -> memref<f32>
  acc.data dataOperands(%5, %6, %7 : memref<f32>, memref<f32>, memref<f32>) {
  }

  %8 = acc.copyin varPtr(%a : memref<f32>) -> memref<f32> {dataClause = #acc<data_clause acc_copyin_readonly>}
  %9 = acc.copyin varPtr(%b : memref<f32>) -> memref<f32> {dataClause = #acc<data_clause acc_copyin_readonly>}
  %10 = acc.copyin varPtr(%c : memref<f32>) -> memref<f32> {dataClause = #acc<data_clause acc_copyin_readonly>}
  acc.data dataOperands(%8, %9, %10 : memref<f32>, memref<f32>, memref<f32>) {
  }

  %11 = acc.create varPtr(%a : memref<f32>) -> memref<f32> {dataClause = #acc<data_clause acc_copyout>}
  %12 = acc.create varPtr(%b : memref<f32>) -> memref<f32> {dataClause = #acc<data_clause acc_copyout>}
  %13 = acc.create varPtr(%c : memref<f32>) -> memref<f32> {dataClause = #acc<data_clause acc_copyout>}
  acc.data dataOperands(%11, %12, %13 : memref<f32>, memref<f32>, memref<f32>) {
  }
  acc.copyout accPtr(%11 : memref<f32>) to varPtr(%a : memref<f32>)
  acc.copyout accPtr(%12 : memref<f32>) to varPtr(%b : memref<f32>)
  acc.copyout accPtr(%13 : memref<f32>) to varPtr(%c : memref<f32>)

  %14 = acc.create varPtr(%a : memref<f32>) -> memref<f32> {dataClause = #acc<data_clause acc_copyout_zero>}
  %15 = acc.create varPtr(%b : memref<f32>) -> memref<f32> {dataClause = #acc<data_clause acc_copyout_zero>}
  %16 = acc.create varPtr(%c : memref<f32>) -> memref<f32> {dataClause = #acc<data_clause acc_copyout_zero>}
  acc.data dataOperands(%14, %15, %16 : memref<f32>, memref<f32>, memref<f32>) {
  }
  acc.copyout accPtr(%14 : memref<f32>) to varPtr(%a : memref<f32>) {dataClause = #acc<data_clause acc_copyout_zero>}
  acc.copyout accPtr(%15 : memref<f32>) to varPtr(%b : memref<f32>) {dataClause = #acc<data_clause acc_copyout_zero>}
  acc.copyout accPtr(%16 : memref<f32>) to varPtr(%c : memref<f32>) {dataClause = #acc<data_clause acc_copyout_zero>}

  %17 = acc.create varPtr(%a : memref<f32>) -> memref<f32>
  %18 = acc.create varPtr(%b : memref<f32>) -> memref<f32>
  %19 = acc.create varPtr(%c : memref<f32>) -> memref<f32>
  acc.data dataOperands(%17, %18, %19 : memref<f32>, memref<f32>, memref<f32>) {
  }
  acc.delete accPtr(%17 : memref<f32>) {dataClause = #acc<data_clause acc_create>}
  acc.delete accPtr(%18 : memref<f32>) {dataClause = #acc<data_clause acc_create>}
  acc.delete accPtr(%19 : memref<f32>) {dataClause = #acc<data_clause acc_create>}
  
  %20 = acc.create varPtr(%a : memref<f32>) -> memref<f32> {dataClause = #acc<data_clause acc_create_zero>}
  %21 = acc.create varPtr(%b : memref<f32>) -> memref<f32> {dataClause = #acc<data_clause acc_create_zero>}
  %22 = acc.create varPtr(%c : memref<f32>) -> memref<f32> {dataClause = #acc<data_clause acc_create_zero>}
  acc.data dataOperands(%20, %21, %22 : memref<f32>, memref<f32>, memref<f32>) {
  }
  acc.delete accPtr(%20 : memref<f32>) {dataClause = #acc<data_clause acc_create_zero>}
  acc.delete accPtr(%21 : memref<f32>) {dataClause = #acc<data_clause acc_create_zero>}
  acc.delete accPtr(%22 : memref<f32>) {dataClause = #acc<data_clause acc_create_zero>}

  %23 = acc.nocreate varPtr(%a : memref<f32>) -> memref<f32>
  %24 = acc.nocreate varPtr(%b : memref<f32>) -> memref<f32>
  %25 = acc.nocreate varPtr(%c : memref<f32>) -> memref<f32>
  acc.data dataOperands(%23, %24, %25 : memref<f32>, memref<f32>, memref<f32>) {
  }

  %26 = acc.deviceptr varPtr(%a : memref<f32>) -> memref<f32>
  %27 = acc.deviceptr varPtr(%b : memref<f32>) -> memref<f32>
  %28 = acc.deviceptr varPtr(%c : memref<f32>) -> memref<f32>
  acc.data dataOperands(%26, %27, %28 : memref<f32>, memref<f32>, memref<f32>) {
  }

  %29 = acc.attach varPtr(%a : memref<f32>) -> memref<f32>
  %30 = acc.attach varPtr(%b : memref<f32>) -> memref<f32>
  %31 = acc.attach varPtr(%c : memref<f32>) -> memref<f32>
  acc.data dataOperands(%29, %30, %31 : memref<f32>, memref<f32>, memref<f32>) {
  }

  %32 = acc.copyin varPtr(%a : memref<f32>) -> memref<f32>
  %33 = acc.create varPtr(%b : memref<f32>) -> memref<f32> {dataClause = #acc<data_clause acc_copyout>}
  %34 = acc.present varPtr(%c : memref<f32>) -> memref<f32>
  acc.data dataOperands(%32, %33, %34 : memref<f32>, memref<f32>, memref<f32>) {
  }
  acc.copyout accPtr(%33 : memref<f32>) to varPtr(%b : memref<f32>)

  %35 = acc.present varPtr(%a : memref<f32>) -> memref<f32>
  acc.data dataOperands(%35 : memref<f32>) {
  } attributes { defaultAttr = #acc<defaultvalue none> }
  

  %36 = acc.present varPtr(%a : memref<f32>) -> memref<f32>
  acc.data dataOperands(%36 : memref<f32>) {
  } attributes { defaultAttr = #acc<defaultvalue present> }

  acc.data {
  } attributes { defaultAttr = #acc<defaultvalue none> }

  acc.data {
  } attributes { defaultAttr = #acc<defaultvalue none>, async }

  %a1 = arith.constant 1 : i64
  acc.data async(%a1 : i64) {
  } attributes { defaultAttr = #acc<defaultvalue none>, async }

  acc.data {
  } attributes { defaultAttr = #acc<defaultvalue none>, wait }

  %w1 = arith.constant 1 : i64
  acc.data wait(%w1 : i64) {
  } attributes { defaultAttr = #acc<defaultvalue none>, wait }

  %wd1 = arith.constant 1 : i64
  acc.data wait_devnum(%wd1 : i64) wait(%w1 : i64) {
  } attributes { defaultAttr = #acc<defaultvalue none>, wait }

  return
}

// CHECK:      func @testdataop(%[[ARGA:.*]]: memref<f32>, %[[ARGB:.*]]: memref<f32>, %[[ARGC:.*]]: memref<f32>) {

// CHECK:      %[[IFCOND1:.*]] = arith.constant true
// CHECK:      %[[PRESENT_A:.*]] = acc.present varPtr(%[[ARGA]] : memref<f32>) -> memref<f32>
// CHECK:      acc.data if(%[[IFCOND1]]) dataOperands(%[[PRESENT_A]] : memref<f32>) {
// CHECK-NEXT: }

// CHECK:      %[[PRESENT_A:.*]] = acc.present varPtr(%[[ARGA]] : memref<f32>) -> memref<f32>
// CHECK:      acc.data if(%[[IFCOND1]]) dataOperands(%[[PRESENT_A]] : memref<f32>) {
// CHECK-NEXT: }

// CHECK:      %[[PRESENT_A:.*]] = acc.present varPtr(%[[ARGA]] : memref<f32>) -> memref<f32>
// CHECK:      %[[PRESENT_B:.*]] = acc.present varPtr(%[[ARGB]] : memref<f32>) -> memref<f32>
// CHECK:      %[[PRESENT_C:.*]] = acc.present varPtr(%[[ARGC]] : memref<f32>) -> memref<f32>
// CHECK:      acc.data dataOperands(%[[PRESENT_A]], %[[PRESENT_B]], %[[PRESENT_C]] : memref<f32>, memref<f32>, memref<f32>) {
// CHECK-NEXT: }

// CHECK:      %[[COPYIN_A:.*]] = acc.copyin varPtr(%[[ARGA]] : memref<f32>) -> memref<f32>
// CHECK:      %[[COPYIN_B:.*]] = acc.copyin varPtr(%[[ARGB]] : memref<f32>) -> memref<f32>
// CHECK:      %[[COPYIN_C:.*]] = acc.copyin varPtr(%[[ARGC]] : memref<f32>) -> memref<f32>
// CHECK:      acc.data dataOperands(%[[COPYIN_A]], %[[COPYIN_B]], %[[COPYIN_C]] : memref<f32>, memref<f32>, memref<f32>) {
// CHECK-NEXT: }

// CHECK:      %[[COPYIN_A:.*]] = acc.copyin varPtr(%[[ARGA]] : memref<f32>) -> memref<f32> {dataClause = #acc<data_clause acc_copyin_readonly>}
// CHECK:      %[[COPYIN_B:.*]] = acc.copyin varPtr(%[[ARGB]] : memref<f32>) -> memref<f32> {dataClause = #acc<data_clause acc_copyin_readonly>}
// CHECK:      %[[COPYIN_C:.*]] = acc.copyin varPtr(%[[ARGC]] : memref<f32>) -> memref<f32> {dataClause = #acc<data_clause acc_copyin_readonly>}
// CHECK:      acc.data dataOperands(%[[COPYIN_A]], %[[COPYIN_B]], %[[COPYIN_C]] : memref<f32>, memref<f32>, memref<f32>) {
// CHECK-NEXT: }

// CHECK:      %[[CREATE_A:.*]] = acc.create varPtr(%[[ARGA]] : memref<f32>) -> memref<f32> {dataClause = #acc<data_clause acc_copyout>}
// CHECK:      %[[CREATE_B:.*]] = acc.create varPtr(%[[ARGB]] : memref<f32>) -> memref<f32> {dataClause = #acc<data_clause acc_copyout>}
// CHECK:      %[[CREATE_C:.*]] = acc.create varPtr(%[[ARGC]] : memref<f32>) -> memref<f32> {dataClause = #acc<data_clause acc_copyout>}
// CHECK:      acc.data dataOperands(%[[CREATE_A]], %[[CREATE_B]], %[[CREATE_C]] : memref<f32>, memref<f32>, memref<f32>) {
// CHECK-NEXT: }
// CHECK:      acc.copyout accPtr(%[[CREATE_A]] : memref<f32>) to varPtr(%[[ARGA]] : memref<f32>)
// CHECK:      acc.copyout accPtr(%[[CREATE_B]] : memref<f32>) to varPtr(%[[ARGB]] : memref<f32>)
// CHECK:      acc.copyout accPtr(%[[CREATE_C]] : memref<f32>) to varPtr(%[[ARGC]] : memref<f32>)

// CHECK:      %[[CREATE_A:.*]] = acc.create varPtr(%[[ARGA]] : memref<f32>) -> memref<f32> {dataClause = #acc<data_clause acc_copyout_zero>}
// CHECK:      %[[CREATE_B:.*]] = acc.create varPtr(%[[ARGB]] : memref<f32>) -> memref<f32> {dataClause = #acc<data_clause acc_copyout_zero>}
// CHECK:      %[[CREATE_C:.*]] = acc.create varPtr(%[[ARGC]] : memref<f32>) -> memref<f32> {dataClause = #acc<data_clause acc_copyout_zero>}
// CHECK:      acc.data dataOperands(%[[CREATE_A]], %[[CREATE_B]], %[[CREATE_C]] : memref<f32>, memref<f32>, memref<f32>) {
// CHECK-NEXT: }
// CHECK:      acc.copyout accPtr(%[[CREATE_A]] : memref<f32>) to varPtr(%[[ARGA]] : memref<f32>) {dataClause = #acc<data_clause acc_copyout_zero>}
// CHECK:      acc.copyout accPtr(%[[CREATE_B]] : memref<f32>) to varPtr(%[[ARGB]] : memref<f32>) {dataClause = #acc<data_clause acc_copyout_zero>}
// CHECK:      acc.copyout accPtr(%[[CREATE_C]] : memref<f32>) to varPtr(%[[ARGC]] : memref<f32>) {dataClause = #acc<data_clause acc_copyout_zero>}

// CHECK:      %[[CREATE_A:.*]] = acc.create varPtr(%[[ARGA]] : memref<f32>) -> memref<f32>
// CHECK:      %[[CREATE_B:.*]] = acc.create varPtr(%[[ARGB]] : memref<f32>) -> memref<f32>
// CHECK:      %[[CREATE_C:.*]] = acc.create varPtr(%[[ARGC]] : memref<f32>) -> memref<f32>
// CHECK:      acc.data dataOperands(%[[CREATE_A]], %[[CREATE_B]], %[[CREATE_C]] : memref<f32>, memref<f32>, memref<f32>) {
// CHECK-NEXT: }

// CHECK:      %[[CREATE_A:.*]] = acc.create varPtr(%[[ARGA]] : memref<f32>) -> memref<f32> {dataClause = #acc<data_clause acc_create_zero>}
// CHECK:      %[[CREATE_B:.*]] = acc.create varPtr(%[[ARGB]] : memref<f32>) -> memref<f32> {dataClause = #acc<data_clause acc_create_zero>}
// CHECK:      %[[CREATE_C:.*]] = acc.create varPtr(%[[ARGC]] : memref<f32>) -> memref<f32> {dataClause = #acc<data_clause acc_create_zero>}
// CHECK:      acc.data dataOperands(%[[CREATE_A]], %[[CREATE_B]], %[[CREATE_C]] : memref<f32>, memref<f32>, memref<f32>) {
// CHECK-NEXT: }

// CHECK:      %[[NOCREATE_A:.*]] = acc.nocreate varPtr(%[[ARGA]] : memref<f32>) -> memref<f32>
// CHECK:      %[[NOCREATE_B:.*]] = acc.nocreate varPtr(%[[ARGB]] : memref<f32>) -> memref<f32>
// CHECK:      %[[NOCREATE_C:.*]] = acc.nocreate varPtr(%[[ARGC]] : memref<f32>) -> memref<f32>
// CHECK:      acc.data dataOperands(%[[NOCREATE_A]], %[[NOCREATE_B]], %[[NOCREATE_C]] : memref<f32>, memref<f32>, memref<f32>) {
// CHECK-NEXT: }

// CHECK:      %[[DEVICEPTR_A:.*]] = acc.deviceptr varPtr(%[[ARGA]] : memref<f32>) -> memref<f32>
// CHECK:      %[[DEVICEPTR_B:.*]] = acc.deviceptr varPtr(%[[ARGB]] : memref<f32>) -> memref<f32>
// CHECK:      %[[DEVICEPTR_C:.*]] = acc.deviceptr varPtr(%[[ARGC]] : memref<f32>) -> memref<f32>
// CHECK:      acc.data dataOperands(%[[DEVICEPTR_A]], %[[DEVICEPTR_B]], %[[DEVICEPTR_C]] : memref<f32>, memref<f32>, memref<f32>) {
// CHECK-NEXT: }

// CHECK:      %[[ATTACH_A:.*]] = acc.attach varPtr(%[[ARGA]] : memref<f32>) -> memref<f32>
// CHECK:      %[[ATTACH_B:.*]] = acc.attach varPtr(%[[ARGB]] : memref<f32>) -> memref<f32>
// CHECK:      %[[ATTACH_C:.*]] = acc.attach varPtr(%[[ARGC]] : memref<f32>) -> memref<f32>
// CHECK:      acc.data dataOperands(%[[ATTACH_A]], %[[ATTACH_B]], %[[ATTACH_C]] : memref<f32>, memref<f32>, memref<f32>) {
// CHECK-NEXT: }


// CHECK:      %[[COPYIN_A:.*]] = acc.copyin varPtr(%[[ARGA]] : memref<f32>) -> memref<f32>
// CHECK:      %[[CREATE_B:.*]] = acc.create varPtr(%[[ARGB]] : memref<f32>) -> memref<f32> {dataClause = #acc<data_clause acc_copyout>}
// CHECK:      %[[PRESENT_C:.*]] = acc.present varPtr(%[[ARGC]] : memref<f32>) -> memref<f32>
// CHECK:      acc.data dataOperands(%[[COPYIN_A]], %[[CREATE_B]], %[[PRESENT_C]] : memref<f32>, memref<f32>, memref<f32>) {
// CHECK-NEXT: }
// CHECK:      acc.copyout accPtr(%[[CREATE_B]] : memref<f32>) to varPtr(%[[ARGB]] : memref<f32>)

// CHECK:      %[[PRESENT_A:.*]] = acc.present varPtr(%[[ARGA]] : memref<f32>) -> memref<f32>
// CHECK:      acc.data dataOperands(%[[PRESENT_A]] : memref<f32>) {
// CHECK-NEXT: } attributes {defaultAttr = #acc<defaultvalue none>}

// CHECK:      %[[PRESENT_A:.*]] = acc.present varPtr(%[[ARGA]] : memref<f32>) -> memref<f32>
// CHECK:      acc.data dataOperands(%[[PRESENT_A]] : memref<f32>) {
// CHECK-NEXT: } attributes {defaultAttr = #acc<defaultvalue present>}

// CHECK:      acc.data {
// CHECK-NEXT: } attributes {defaultAttr = #acc<defaultvalue none>}

// CHECK:      acc.data {
// CHECK-NEXT: } attributes {async, defaultAttr = #acc<defaultvalue none>}

// CHECK:      acc.data async(%{{.*}} : i64) {
// CHECK-NEXT: } attributes {async, defaultAttr = #acc<defaultvalue none>}

// CHECK:      acc.data {
// CHECK-NEXT: } attributes {defaultAttr = #acc<defaultvalue none>, wait}

// CHECK:      acc.data wait(%{{.*}} : i64) {
// CHECK-NEXT: } attributes {defaultAttr = #acc<defaultvalue none>, wait}

// CHECK:      acc.data wait_devnum(%{{.*}} : i64) wait(%{{.*}} : i64) {
// CHECK-NEXT: } attributes {defaultAttr = #acc<defaultvalue none>, wait}

// -----

func.func @testupdateop(%a: memref<f32>, %b: memref<f32>, %c: memref<f32>) -> () {
  %i64Value = arith.constant 1 : i64
  %i32Value = arith.constant 1 : i32
  %idxValue = arith.constant 1 : index
  %ifCond = arith.constant true
  %0 = acc.update_device varPtr(%a : memref<f32>) -> memref<f32>
  %1 = acc.update_device varPtr(%b : memref<f32>) -> memref<f32>
  %2 = acc.update_device varPtr(%c : memref<f32>) -> memref<f32>
  
  acc.update async(%i64Value: i64) dataOperands(%0: memref<f32>)
  acc.update async(%i32Value: i32) dataOperands(%0: memref<f32>)
  acc.update async(%i32Value: i32) dataOperands(%0: memref<f32>)
  acc.update async(%idxValue: index) dataOperands(%0: memref<f32>)
  acc.update wait_devnum(%i64Value: i64) wait(%i32Value, %idxValue : i32, index) dataOperands(%0: memref<f32>)
  acc.update if(%ifCond) dataOperands(%0: memref<f32>)
  acc.update dataOperands(%0: memref<f32>) attributes {acc.device_types = [#acc.device_type<star>]}
  acc.update dataOperands(%0, %1, %2 : memref<f32>, memref<f32>, memref<f32>)
  acc.update dataOperands(%0, %1, %2 : memref<f32>, memref<f32>, memref<f32>) attributes {async}
  acc.update dataOperands(%0, %1, %2 : memref<f32>, memref<f32>, memref<f32>) attributes {wait}
  acc.update dataOperands(%0, %1, %2 : memref<f32>, memref<f32>, memref<f32>) attributes {ifPresent}
  return
}

// CHECK: func @testupdateop([[ARGA:%.*]]: memref<f32>, [[ARGB:%.*]]: memref<f32>, [[ARGC:%.*]]: memref<f32>) {
// CHECK:   [[I64VALUE:%.*]] = arith.constant 1 : i64
// CHECK:   [[I32VALUE:%.*]] = arith.constant 1 : i32
// CHECK:   [[IDXVALUE:%.*]] = arith.constant 1 : index
// CHECK:   [[IFCOND:%.*]] = arith.constant true
// CHECK:   acc.update async([[I64VALUE]] : i64) dataOperands(%{{.*}} : memref<f32>)
// CHECK:   acc.update async([[I32VALUE]] : i32) dataOperands(%{{.*}} : memref<f32>)
// CHECK:   acc.update async([[I32VALUE]] : i32) dataOperands(%{{.*}} : memref<f32>)
// CHECK:   acc.update async([[IDXVALUE]] : index) dataOperands(%{{.*}} : memref<f32>)
// CHECK:   acc.update wait_devnum([[I64VALUE]] : i64) wait([[I32VALUE]], [[IDXVALUE]] : i32, index) dataOperands(%{{.*}} : memref<f32>)
// CHECK:   acc.update if([[IFCOND]]) dataOperands(%{{.*}} : memref<f32>)
// CHECK:   acc.update dataOperands(%{{.*}} : memref<f32>) attributes {acc.device_types = [#acc.device_type<star>]}
// CHECK:   acc.update dataOperands(%{{.*}}, %{{.*}}, %{{.*}} : memref<f32>, memref<f32>, memref<f32>)
// CHECK:   acc.update dataOperands(%{{.*}}, %{{.*}}, %{{.*}} : memref<f32>, memref<f32>, memref<f32>) attributes {async}
// CHECK:   acc.update dataOperands(%{{.*}}, %{{.*}}, %{{.*}} : memref<f32>, memref<f32>, memref<f32>) attributes {wait}
// CHECK:   acc.update dataOperands(%{{.*}}, %{{.*}}, %{{.*}} : memref<f32>, memref<f32>, memref<f32>) attributes {ifPresent}

// -----

%i64Value = arith.constant 1 : i64
%i32Value = arith.constant 1 : i32
%idxValue = arith.constant 1 : index
%ifCond = arith.constant true
acc.wait
acc.wait(%i64Value: i64)
acc.wait(%i32Value: i32)
acc.wait(%idxValue: index)
acc.wait(%i32Value, %idxValue : i32, index)
acc.wait async(%i64Value: i64)
acc.wait async(%i32Value: i32)
acc.wait async(%idxValue: index)
acc.wait(%i32Value: i32) async(%idxValue: index)
acc.wait(%i64Value: i64) wait_devnum(%i32Value: i32)
acc.wait attributes {async}
acc.wait(%i64Value: i64) async(%idxValue: index) wait_devnum(%i32Value: i32)
acc.wait(%i64Value: i64) wait_devnum(%i32Value: i32) async(%idxValue: index)
acc.wait if(%ifCond)

// CHECK: [[I64VALUE:%.*]] = arith.constant 1 : i64
// CHECK: [[I32VALUE:%.*]] = arith.constant 1 : i32
// CHECK: [[IDXVALUE:%.*]] = arith.constant 1 : index
// CHECK: [[IFCOND:%.*]] = arith.constant true
// CHECK: acc.wait
// CHECK: acc.wait([[I64VALUE]] : i64)
// CHECK: acc.wait([[I32VALUE]] : i32)
// CHECK: acc.wait([[IDXVALUE]] : index)
// CHECK: acc.wait([[I32VALUE]], [[IDXVALUE]] : i32, index)
// CHECK: acc.wait async([[I64VALUE]] : i64)
// CHECK: acc.wait async([[I32VALUE]] : i32)
// CHECK: acc.wait async([[IDXVALUE]] : index)
// CHECK: acc.wait([[I32VALUE]] : i32) async([[IDXVALUE]] : index)
// CHECK: acc.wait([[I64VALUE]] : i64) wait_devnum([[I32VALUE]] : i32)
// CHECK: acc.wait attributes {async}
// CHECK: acc.wait([[I64VALUE]] : i64) async([[IDXVALUE]] : index) wait_devnum([[I32VALUE]] : i32)
// CHECK: acc.wait([[I64VALUE]] : i64) async([[IDXVALUE]] : index) wait_devnum([[I32VALUE]] : i32)
// CHECK: acc.wait if([[IFCOND]])

// -----

%i64Value = arith.constant 1 : i64
%i32Value = arith.constant 1 : i32
%i32Value2 = arith.constant 2 : i32
%idxValue = arith.constant 1 : index
%ifCond = arith.constant true
acc.init
acc.init attributes {acc.device_types = [#acc.device_type<nvidia>]}
acc.init device_num(%i64Value : i64)
acc.init device_num(%i32Value : i32)
acc.init device_num(%idxValue : index)
acc.init if(%ifCond)
acc.init if(%ifCond) device_num(%idxValue : index)
acc.init device_num(%idxValue : index) if(%ifCond)

// CHECK: [[I64VALUE:%.*]] = arith.constant 1 : i64
// CHECK: [[I32VALUE:%.*]] = arith.constant 1 : i32
// CHECK: [[I32VALUE2:%.*]] = arith.constant 2 : i32
// CHECK: [[IDXVALUE:%.*]] = arith.constant 1 : index
// CHECK: [[IFCOND:%.*]] = arith.constant true
// CHECK: acc.init
// CHECK: acc.init attributes {acc.device_types = [#acc.device_type<nvidia>]} 
// CHECK: acc.init device_num([[I64VALUE]] : i64)
// CHECK: acc.init device_num([[I32VALUE]] : i32)
// CHECK: acc.init device_num([[IDXVALUE]] : index)
// CHECK: acc.init if([[IFCOND]])
// CHECK: acc.init device_num([[IDXVALUE]] : index) if([[IFCOND]])
// CHECK: acc.init device_num([[IDXVALUE]] : index) if([[IFCOND]])

// -----

%i64Value = arith.constant 1 : i64
%i32Value = arith.constant 1 : i32
%i32Value2 = arith.constant 2 : i32
%idxValue = arith.constant 1 : index
%ifCond = arith.constant true
acc.shutdown
acc.shutdown attributes {acc.device_types = [#acc.device_type<default>]}
acc.shutdown device_num(%i64Value : i64)
acc.shutdown device_num(%i32Value : i32)
acc.shutdown device_num(%idxValue : index)
acc.shutdown if(%ifCond)
acc.shutdown if(%ifCond) device_num(%idxValue : index)
acc.shutdown device_num(%idxValue : index) if(%ifCond)

// CHECK: [[I64VALUE:%.*]] = arith.constant 1 : i64
// CHECK: [[I32VALUE:%.*]] = arith.constant 1 : i32
// CHECK: [[I32VALUE2:%.*]] = arith.constant 2 : i32
// CHECK: [[IDXVALUE:%.*]] = arith.constant 1 : index
// CHECK: [[IFCOND:%.*]] = arith.constant true
// CHECK: acc.shutdown
// CHECK: acc.shutdown attributes {acc.device_types = [#acc.device_type<default>]}
// CHECK: acc.shutdown device_num([[I64VALUE]] : i64)
// CHECK: acc.shutdown device_num([[I32VALUE]] : i32)
// CHECK: acc.shutdown device_num([[IDXVALUE]] : index)
// CHECK: acc.shutdown if([[IFCOND]])
// CHECK: acc.shutdown device_num([[IDXVALUE]] : index) if([[IFCOND]])
// CHECK: acc.shutdown device_num([[IDXVALUE]] : index) if([[IFCOND]])

// -----

func.func @testexitdataop(%a: !llvm.ptr) -> () {
  %ifCond = arith.constant true
  %i64Value = arith.constant 1 : i64
  %i32Value = arith.constant 1 : i32
  %idxValue = arith.constant 1 : index

  %0 = acc.getdeviceptr varPtr(%a : !llvm.ptr) -> !llvm.ptr
  acc.exit_data dataOperands(%0 : !llvm.ptr)
  acc.copyout accPtr(%0 : !llvm.ptr) to varPtr(%a : !llvm.ptr)

  %1 = acc.getdeviceptr varPtr(%a : !llvm.ptr) -> !llvm.ptr
  acc.exit_data dataOperands(%1 : !llvm.ptr)
  acc.delete accPtr(%1 : !llvm.ptr)

  %2 = acc.getdeviceptr varPtr(%a : !llvm.ptr) -> !llvm.ptr
  acc.exit_data dataOperands(%2 : !llvm.ptr) attributes {async,finalize}
  acc.delete accPtr(%2 : !llvm.ptr)

  %3 = acc.getdeviceptr varPtr(%a : !llvm.ptr) -> !llvm.ptr
  acc.exit_data dataOperands(%3 : !llvm.ptr)
  acc.detach accPtr(%3 : !llvm.ptr)

  %4 = acc.getdeviceptr varPtr(%a : !llvm.ptr) -> !llvm.ptr
  acc.exit_data dataOperands(%4 : !llvm.ptr) attributes {async}
  acc.copyout accPtr(%4 : !llvm.ptr) to varPtr(%a : !llvm.ptr)

  %5 = acc.getdeviceptr varPtr(%a : !llvm.ptr) -> !llvm.ptr
  acc.exit_data dataOperands(%5 : !llvm.ptr) attributes {wait}
  acc.delete accPtr(%5 : !llvm.ptr)

  %6 = acc.getdeviceptr varPtr(%a : !llvm.ptr) -> !llvm.ptr
  acc.exit_data async(%i64Value : i64) dataOperands(%6 : !llvm.ptr)
  acc.copyout accPtr(%6 : !llvm.ptr) to varPtr(%a : !llvm.ptr)

  %7 = acc.getdeviceptr varPtr(%a : !llvm.ptr) -> !llvm.ptr
  acc.exit_data dataOperands(%7 : !llvm.ptr) async(%i64Value : i64)
  acc.copyout accPtr(%7 : !llvm.ptr) to varPtr(%a : !llvm.ptr)

  %8 = acc.getdeviceptr varPtr(%a : !llvm.ptr) -> !llvm.ptr
  acc.exit_data if(%ifCond) dataOperands(%8 : !llvm.ptr)
  acc.copyout accPtr(%8 : !llvm.ptr) to varPtr(%a : !llvm.ptr)

  %9 = acc.getdeviceptr varPtr(%a : !llvm.ptr) -> !llvm.ptr
  acc.exit_data wait_devnum(%i64Value: i64) wait(%i32Value, %idxValue : i32, index) dataOperands(%9 : !llvm.ptr)
  acc.copyout accPtr(%9 : !llvm.ptr) to varPtr(%a : !llvm.ptr)

  return
}

// CHECK: func @testexitdataop(%[[ARGA:.*]]: !llvm.ptr) {
// CHECK: %[[IFCOND:.*]] = arith.constant true
// CHECK: %[[I64VALUE:.*]] = arith.constant 1 : i64
// CHECK: %[[I32VALUE:.*]] = arith.constant 1 : i32
// CHECK: %[[IDXVALUE:.*]] = arith.constant 1 : index

// CHECK: %[[DEVPTR:.*]] = acc.getdeviceptr varPtr(%[[ARGA]] : !llvm.ptr) -> !llvm.ptr
// CHECK: acc.exit_data dataOperands(%[[DEVPTR]] : !llvm.ptr)
// CHECK: acc.copyout accPtr(%[[DEVPTR]] : !llvm.ptr) to varPtr(%[[ARGA]] : !llvm.ptr)

// CHECK: %[[DEVPTR:.*]] = acc.getdeviceptr varPtr(%[[ARGA]] : !llvm.ptr) -> !llvm.ptr
// CHECK: acc.exit_data dataOperands(%[[DEVPTR]] : !llvm.ptr)
// CHECK: acc.delete accPtr(%[[DEVPTR]] : !llvm.ptr)

// CHECK: %[[DEVPTR:.*]] = acc.getdeviceptr varPtr(%[[ARGA]] : !llvm.ptr) -> !llvm.ptr
// CHECK: acc.exit_data dataOperands(%[[DEVPTR]] : !llvm.ptr) attributes {async, finalize}
// CHECK: acc.delete accPtr(%[[DEVPTR]] : !llvm.ptr)

// CHECK: %[[DEVPTR:.*]] = acc.getdeviceptr varPtr(%[[ARGA]] : !llvm.ptr) -> !llvm.ptr
// CHECK: acc.exit_data dataOperands(%[[DEVPTR]] : !llvm.ptr)
// CHECK: acc.detach accPtr(%[[DEVPTR]] : !llvm.ptr)

// CHECK: %[[DEVPTR:.*]] = acc.getdeviceptr varPtr(%[[ARGA]] : !llvm.ptr) -> !llvm.ptr
// CHECK: acc.exit_data dataOperands(%[[DEVPTR]] : !llvm.ptr) attributes {async}
// CHECK: acc.copyout accPtr(%[[DEVPTR]] : !llvm.ptr) to varPtr(%[[ARGA]] : !llvm.ptr)

// CHECK: %[[DEVPTR:.*]] = acc.getdeviceptr varPtr(%[[ARGA]] : !llvm.ptr) -> !llvm.ptr
// CHECK: acc.exit_data dataOperands(%[[DEVPTR]] : !llvm.ptr) attributes {wait}
// CHECK: acc.delete accPtr(%[[DEVPTR]] : !llvm.ptr)

// CHECK: %[[DEVPTR:.*]] = acc.getdeviceptr varPtr(%[[ARGA]] : !llvm.ptr) -> !llvm.ptr
// CHECK: acc.exit_data async(%[[I64VALUE]] : i64) dataOperands(%[[DEVPTR]] : !llvm.ptr)
// CHECK: acc.copyout accPtr(%[[DEVPTR]] : !llvm.ptr) to varPtr(%[[ARGA]] : !llvm.ptr)

// CHECK: %[[DEVPTR:.*]] = acc.getdeviceptr varPtr(%[[ARGA]] : !llvm.ptr) -> !llvm.ptr
// CHECK: acc.exit_data async(%[[I64VALUE]] : i64) dataOperands(%[[DEVPTR]] : !llvm.ptr)
// CHECK: acc.copyout accPtr(%[[DEVPTR]] : !llvm.ptr) to varPtr(%[[ARGA]] : !llvm.ptr)

// CHECK: %[[DEVPTR:.*]] = acc.getdeviceptr varPtr(%[[ARGA]] : !llvm.ptr) -> !llvm.ptr
// CHECK: acc.exit_data if(%[[IFCOND]]) dataOperands(%[[DEVPTR]] : !llvm.ptr)
// CHECK: acc.copyout accPtr(%[[DEVPTR]] : !llvm.ptr) to varPtr(%[[ARGA]] : !llvm.ptr)

// CHECK: %[[DEVPTR:.*]] = acc.getdeviceptr varPtr(%[[ARGA]] : !llvm.ptr) -> !llvm.ptr
// CHECK: acc.exit_data wait_devnum(%[[I64VALUE]] : i64) wait(%[[I32VALUE]], %[[IDXVALUE]] : i32, index) dataOperands(%[[DEVPTR]] : !llvm.ptr)
// CHECK: acc.copyout accPtr(%[[DEVPTR]] : !llvm.ptr) to varPtr(%[[ARGA]] : !llvm.ptr)

// -----


func.func @testenterdataop(%a: !llvm.ptr, %b: !llvm.ptr, %c: !llvm.ptr) -> () {
  %ifCond = arith.constant true
  %i64Value = arith.constant 1 : i64
  %i32Value = arith.constant 1 : i32
  %idxValue = arith.constant 1 : index

  %0 = acc.copyin varPtr(%a : !llvm.ptr) -> !llvm.ptr
  acc.enter_data dataOperands(%0 : !llvm.ptr)
  %1 = acc.create varPtr(%a : !llvm.ptr) -> !llvm.ptr
  %2 = acc.create varPtr(%b : !llvm.ptr) -> !llvm.ptr {dataClause = #acc<data_clause acc_create_zero>}
  %3 = acc.create varPtr(%c : !llvm.ptr) -> !llvm.ptr {dataClause = #acc<data_clause acc_create_zero>}
  acc.enter_data dataOperands(%1, %2, %3 : !llvm.ptr, !llvm.ptr, !llvm.ptr)
  %4 = acc.attach varPtr(%a : !llvm.ptr) -> !llvm.ptr
  acc.enter_data dataOperands(%4 : !llvm.ptr)
  %5 = acc.copyin varPtr(%a : !llvm.ptr) -> !llvm.ptr
  acc.enter_data dataOperands(%5 : !llvm.ptr) attributes {async}
  %6 = acc.create varPtr(%a : !llvm.ptr) -> !llvm.ptr
  acc.enter_data dataOperands(%6 : !llvm.ptr) attributes {wait}
  %7 = acc.copyin varPtr(%a : !llvm.ptr) -> !llvm.ptr
  acc.enter_data async(%i64Value : i64) dataOperands(%7 : !llvm.ptr)
  %8 = acc.copyin varPtr(%a : !llvm.ptr) -> !llvm.ptr
  acc.enter_data dataOperands(%8 : !llvm.ptr) async(%i64Value : i64)
  %9 = acc.copyin varPtr(%a : !llvm.ptr) -> !llvm.ptr
  acc.enter_data if(%ifCond) dataOperands(%9 : !llvm.ptr)
  %10 = acc.copyin varPtr(%a : !llvm.ptr) -> !llvm.ptr
  acc.enter_data wait_devnum(%i64Value: i64) wait(%i32Value, %idxValue : i32, index) dataOperands(%10 : !llvm.ptr)

  return
}

// CHECK: func @testenterdataop(%[[ARGA:.*]]: !llvm.ptr, %[[ARGB:.*]]: !llvm.ptr, %[[ARGC:.*]]: !llvm.ptr) {
// CHECK: [[IFCOND:%.*]] = arith.constant true
// CHECK: [[I64VALUE:%.*]] = arith.constant 1 : i64
// CHECK: [[I32VALUE:%.*]] = arith.constant 1 : i32
// CHECK: [[IDXVALUE:%.*]] = arith.constant 1 : index
// CHECK: %[[COPYIN:.*]] = acc.copyin varPtr(%[[ARGA]] : !llvm.ptr) -> !llvm.ptr
// CHECK: acc.enter_data dataOperands(%[[COPYIN]] : !llvm.ptr)
// CHECK: %[[CREATE_A:.*]] = acc.create varPtr(%[[ARGA]] : !llvm.ptr) -> !llvm.ptr
// CHECK: %[[CREATE_B:.*]] = acc.create varPtr(%[[ARGB]] : !llvm.ptr) -> !llvm.ptr {dataClause = #acc<data_clause acc_create_zero>}
// CHECK: %[[CREATE_C:.*]] = acc.create varPtr(%[[ARGC]] : !llvm.ptr) -> !llvm.ptr {dataClause = #acc<data_clause acc_create_zero>}
// CHECK: acc.enter_data dataOperands(%[[CREATE_A]], %[[CREATE_B]], %[[CREATE_C]] : !llvm.ptr, !llvm.ptr, !llvm.ptr)
// CHECK: %[[ATTACH:.*]] = acc.attach varPtr(%[[ARGA]] : !llvm.ptr) -> !llvm.ptr
// CHECK: acc.enter_data dataOperands(%[[ATTACH]] : !llvm.ptr)
// CHECK: %[[COPYIN:.*]] = acc.copyin varPtr(%[[ARGA]] : !llvm.ptr) -> !llvm.ptr
// CHECK: acc.enter_data dataOperands(%[[COPYIN]] : !llvm.ptr) attributes {async}
// CHECK: %[[CREATE:.*]] = acc.create varPtr(%[[ARGA]] : !llvm.ptr) -> !llvm.ptr
// CHECK: acc.enter_data dataOperands(%[[CREATE]] : !llvm.ptr) attributes {wait}
// CHECK: %[[COPYIN:.*]] = acc.copyin varPtr(%[[ARGA]] : !llvm.ptr) -> !llvm.ptr
// CHECK: acc.enter_data async([[I64VALUE]] : i64) dataOperands(%[[COPYIN]] : !llvm.ptr)
// CHECK: %[[COPYIN:.*]] = acc.copyin varPtr(%[[ARGA]] : !llvm.ptr) -> !llvm.ptr
// CHECK: acc.enter_data async([[I64VALUE]] : i64) dataOperands(%[[COPYIN]] : !llvm.ptr)
// CHECK: %[[COPYIN:.*]] = acc.copyin varPtr(%[[ARGA]] : !llvm.ptr) -> !llvm.ptr
// CHECK: acc.enter_data if([[IFCOND]]) dataOperands(%[[COPYIN]] : !llvm.ptr)
// CHECK: %[[COPYIN:.*]] = acc.copyin varPtr(%[[ARGA]] : !llvm.ptr) -> !llvm.ptr
// CHECK: acc.enter_data wait_devnum([[I64VALUE]] : i64) wait([[I32VALUE]], [[IDXVALUE]] : i32, index) dataOperands(%[[COPYIN]] : !llvm.ptr)

// -----

func.func @teststructureddataclauseops(%a: memref<10xf32>, %b: memref<memref<10xf32>>, %c: memref<10x20xf32>) -> () {
  %deviceptr = acc.deviceptr varPtr(%a : memref<10xf32>) -> memref<10xf32> {name = "arrayA"}
  acc.parallel dataOperands(%deviceptr : memref<10xf32>) {
  }

  %present = acc.present varPtr(%a : memref<10xf32>) -> memref<10xf32>
  acc.data dataOperands(%present : memref<10xf32>) {
  }

  %copyin = acc.copyin varPtr(%a : memref<10xf32>) -> memref<10xf32>
  acc.parallel dataOperands(%copyin : memref<10xf32>) {
  }

  %copyinreadonly = acc.copyin varPtr(%a : memref<10xf32>) -> memref<10xf32> {dataClause = #acc<data_clause acc_copyin_readonly>}
  acc.kernels dataOperands(%copyinreadonly : memref<10xf32>) {
  }

  %copyinfromcopy = acc.copyin varPtr(%a : memref<10xf32>) -> memref<10xf32> {dataClause = #acc<data_clause acc_copy>}
  acc.serial dataOperands(%copyinfromcopy : memref<10xf32>) {
  }
  acc.copyout accPtr(%copyinfromcopy : memref<10xf32>) to varPtr(%a : memref<10xf32>) {dataClause = #acc<data_clause acc_copy>}

  %create = acc.create varPtr(%a : memref<10xf32>) -> memref<10xf32>
  %createimplicit = acc.create varPtr(%c : memref<10x20xf32>) -> memref<10x20xf32> {implicit = true}
  acc.parallel dataOperands(%create, %createimplicit : memref<10xf32>, memref<10x20xf32>) {
  }
  acc.delete accPtr(%create : memref<10xf32>) {dataClause = #acc<data_clause acc_create>}
  acc.delete accPtr(%createimplicit : memref<10x20xf32>) {dataClause = #acc<data_clause acc_create>, implicit = true}

  %copyoutzero = acc.create varPtr(%a : memref<10xf32>) -> memref<10xf32> {dataClause = #acc<data_clause acc_copyout_zero>}
  acc.parallel dataOperands(%copyoutzero: memref<10xf32>) {
  }
  acc.copyout accPtr(%copyoutzero : memref<10xf32>) to varPtr(%a : memref<10xf32>) {dataClause = #acc<data_clause acc_copyout_zero>}

  %attach = acc.attach varPtr(%b : memref<memref<10xf32>>) -> memref<memref<10xf32>>
  acc.parallel dataOperands(%attach : memref<memref<10xf32>>) {
  }
  acc.detach accPtr(%attach : memref<memref<10xf32>>) {dataClause = #acc<data_clause acc_attach>}

  %copyinparent = acc.copyin varPtr(%a : memref<10xf32>) varPtrPtr(%b : memref<memref<10xf32>>) -> memref<10xf32> {dataClause = #acc<data_clause acc_copy>}
  acc.parallel dataOperands(%copyinparent : memref<10xf32>) {
  }
  acc.copyout accPtr(%copyinparent : memref<10xf32>) to varPtr(%a : memref<10xf32>) {dataClause = #acc<data_clause acc_copy>}

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c9 = arith.constant 9 : index
  %c10 = arith.constant 10 : index
  %c20 = arith.constant 20 : index

  %bounds1full = acc.bounds lowerbound(%c0 : index) upperbound(%c9 : index) stride(%c1 : index)
  %copyinfullslice1 = acc.copyin varPtr(%a : memref<10xf32>) bounds(%bounds1full) -> memref<10xf32> {name = "arrayA[0:9]"}
  // Specify full-bounds but assume that startIdx of array reference is 1.
  %bounds2full = acc.bounds lowerbound(%c1 : index) upperbound(%c20 : index) extent(%c20 : index) stride(%c4 : index) startIdx(%c1 : index) {strideInBytes = true}
  %copyinfullslice2 = acc.copyin varPtr(%c : memref<10x20xf32>) bounds(%bounds1full, %bounds2full) -> memref<10x20xf32>
  acc.parallel dataOperands(%copyinfullslice1, %copyinfullslice2 : memref<10xf32>, memref<10x20xf32>) {
  }

  %bounds1partial = acc.bounds lowerbound(%c4 : index) upperbound(%c9 : index) stride(%c1 : index)
  %copyinpartial = acc.copyin varPtr(%a : memref<10xf32>) bounds(%bounds1partial) -> memref<10xf32> {dataClause = #acc<data_clause acc_copy>}
  acc.parallel dataOperands(%copyinpartial : memref<10xf32>) {
  }
  acc.copyout accPtr(%copyinpartial : memref<10xf32>) bounds(%bounds1partial) to varPtr(%a : memref<10xf32>) {dataClause = #acc<data_clause acc_copy>}

  return
}

// CHECK: func.func @teststructureddataclauseops([[ARGA:%.*]]: memref<10xf32>, [[ARGB:%.*]]: memref<memref<10xf32>>, [[ARGC:%.*]]: memref<10x20xf32>) {
// CHECK: [[DEVICEPTR:%.*]] = acc.deviceptr varPtr([[ARGA]] : memref<10xf32>) -> memref<10xf32> {name = "arrayA"}
// CHECK-NEXT: acc.parallel dataOperands([[DEVICEPTR]] : memref<10xf32>) {
// CHECK-NEXT: }
// CHECK: [[PRESENT:%.*]] = acc.present varPtr([[ARGA]] : memref<10xf32>) -> memref<10xf32>
// CHECK-NEXT: acc.data dataOperands([[PRESENT]] : memref<10xf32>) {
// CHECK-NEXT: }
// CHECK: [[COPYIN:%.*]] = acc.copyin varPtr([[ARGA]] : memref<10xf32>) -> memref<10xf32>
// CHECK-NEXT: acc.parallel dataOperands([[COPYIN]] : memref<10xf32>) {
// CHECK-NEXT: }
// CHECK: [[COPYINRO:%.*]] = acc.copyin varPtr([[ARGA]] : memref<10xf32>) -> memref<10xf32> {dataClause = #acc<data_clause acc_copyin_readonly>}
// CHECK-NEXT: acc.kernels dataOperands([[COPYINRO]] : memref<10xf32>) {
// CHECK-NEXT: }
// CHECK: [[COPYINCOPY:%.*]] = acc.copyin varPtr([[ARGA]] : memref<10xf32>) -> memref<10xf32> {dataClause = #acc<data_clause acc_copy>}
// CHECK-NEXT: acc.serial dataOperands([[COPYINCOPY]] : memref<10xf32>) {
// CHECK-NEXT: }
// CHECK-NEXT: acc.copyout accPtr([[COPYINCOPY]] : memref<10xf32>) to varPtr([[ARGA]] : memref<10xf32>) {dataClause = #acc<data_clause acc_copy>}
// CHECK: [[CREATE:%.*]] = acc.create varPtr([[ARGA]] : memref<10xf32>) -> memref<10xf32>
// CHECK-NEXT: [[CREATEIMP:%.*]] = acc.create varPtr([[ARGC]] : memref<10x20xf32>) -> memref<10x20xf32> {implicit = true}
// CHECK-NEXT: acc.parallel dataOperands([[CREATE]], [[CREATEIMP]] : memref<10xf32>, memref<10x20xf32>) {
// CHECK-NEXT: }
// CHECK-NEXT: acc.delete accPtr([[CREATE]] : memref<10xf32>) {dataClause = #acc<data_clause acc_create>}
// CHECK-NEXT: acc.delete accPtr([[CREATEIMP]] : memref<10x20xf32>) {dataClause = #acc<data_clause acc_create>, implicit = true}
// CHECK: [[COPYOUTZ:%.*]] = acc.create varPtr([[ARGA]] : memref<10xf32>) -> memref<10xf32> {dataClause = #acc<data_clause acc_copyout_zero>}
// CHECK-NEXT: acc.parallel dataOperands([[COPYOUTZ]] : memref<10xf32>) {
// CHECK-NEXT: }
// CHECK-NEXT: acc.copyout accPtr([[COPYOUTZ]] : memref<10xf32>) to varPtr([[ARGA]] : memref<10xf32>) {dataClause = #acc<data_clause acc_copyout_zero>}
// CHECK: [[ATTACH:%.*]] = acc.attach varPtr([[ARGB]] : memref<memref<10xf32>>) -> memref<memref<10xf32>>
// CHECK-NEXT: acc.parallel dataOperands([[ATTACH]] : memref<memref<10xf32>>) {
// CHECK-NEXT: }
// CHECK-NEXT: acc.detach accPtr([[ATTACH]] : memref<memref<10xf32>>) {dataClause = #acc<data_clause acc_attach>}
// CHECK: [[COPYINP:%.*]] = acc.copyin varPtr([[ARGA]] : memref<10xf32>) varPtrPtr([[ARGB]] : memref<memref<10xf32>>) -> memref<10xf32> {dataClause = #acc<data_clause acc_copy>}
// CHECK-NEXT: acc.parallel dataOperands([[COPYINP]] : memref<10xf32>) {
// CHECK-NEXT: }
// CHECK-NEXT: acc.copyout accPtr([[COPYINP]] : memref<10xf32>) to varPtr([[ARGA]] : memref<10xf32>) {dataClause = #acc<data_clause acc_copy>}
// CHECK-DAG: [[CON0:%.*]] = arith.constant 0 : index
// CHECK-DAG: [[CON1:%.*]] = arith.constant 1 : index
// CHECK-DAG: [[CON4:%.*]] = arith.constant 4 : index
// CHECK-DAG: [[CON9:%.*]] = arith.constant 9 : index
// CHECK-DAG: [[CON20:%.*]] = arith.constant 20 : index
// CHECK: [[BOUNDS1F:%.*]] = acc.bounds lowerbound([[CON0]] : index) upperbound([[CON9]] : index) stride([[CON1]] : index)
// CHECK-NEXT: [[COPYINF1:%.*]] = acc.copyin varPtr([[ARGA]] : memref<10xf32>) bounds([[BOUNDS1F]]) -> memref<10xf32> {name = "arrayA[0:9]"}
// CHECK-NEXT: [[BOUNDS2F:%.*]] = acc.bounds lowerbound([[CON1]] : index) upperbound([[CON20]] : index) extent([[CON20]] : index) stride([[CON4]] : index) startIdx([[CON1]] : index) {strideInBytes = true}
// CHECK-NEXT: [[COPYINF2:%.*]] = acc.copyin varPtr([[ARGC]] : memref<10x20xf32>) bounds([[BOUNDS1F]], [[BOUNDS2F]]) -> memref<10x20xf32>
// CHECK-NEXT: acc.parallel dataOperands([[COPYINF1]], [[COPYINF2]] : memref<10xf32>, memref<10x20xf32>) {
// CHECK-NEXT: }
// CHECK: [[BOUNDS1P:%.*]] = acc.bounds lowerbound([[CON4]] : index) upperbound([[CON9]] : index) stride([[CON1]] : index)
// CHECK-NEXT: [[COPYINPART:%.*]] = acc.copyin varPtr([[ARGA]] : memref<10xf32>) bounds([[BOUNDS1P]]) -> memref<10xf32> {dataClause = #acc<data_clause acc_copy>}
// CHECK-NEXT: acc.parallel dataOperands([[COPYINPART]] : memref<10xf32>) {
// CHECK-NEXT: }
// CHECK-NEXT: acc.copyout accPtr([[COPYINPART]] : memref<10xf32>) bounds([[BOUNDS1P]]) to varPtr([[ARGA]] : memref<10xf32>) {dataClause = #acc<data_clause acc_copy>}

// -----

func.func @testunstructuredclauseops(%a: memref<10xf32>) -> () {
  %copyin = acc.copyin varPtr(%a : memref<10xf32>) -> memref<10xf32> {structured = false}
  acc.enter_data dataOperands(%copyin : memref<10xf32>)

  %devptr = acc.getdeviceptr varPtr(%a : memref<10xf32>) -> memref<10xf32> {dataClause = #acc<data_clause acc_copyout>}
  acc.exit_data dataOperands(%devptr : memref<10xf32>)
  acc.copyout accPtr(%devptr : memref<10xf32>) to varPtr(%a : memref<10xf32>) {structured = false}

  return
}

// CHECK: func.func @testunstructuredclauseops([[ARGA:%.*]]: memref<10xf32>) {
// CHECK: [[COPYIN:%.*]] = acc.copyin varPtr([[ARGA]] : memref<10xf32>) -> memref<10xf32> {structured = false}
// CHECK-NEXT: acc.enter_data dataOperands([[COPYIN]] : memref<10xf32>)
// CHECK: [[DEVPTR:%.*]] = acc.getdeviceptr varPtr([[ARGA]] : memref<10xf32>) -> memref<10xf32> {dataClause = #acc<data_clause acc_copyout>}
// CHECK-NEXT: acc.exit_data dataOperands([[DEVPTR]] : memref<10xf32>)
// CHECK-NEXT: acc.copyout accPtr([[DEVPTR]] : memref<10xf32>) to varPtr([[ARGA]] : memref<10xf32>) {structured = false}

// -----

func.func @host_device_ops(%a: memref<f32>) -> () {
  %devptr = acc.getdeviceptr varPtr(%a : memref<f32>) -> memref<f32>
  acc.update_host accPtr(%devptr : memref<f32>) to varPtr(%a : memref<f32>) {structured = false}
  acc.update dataOperands(%devptr : memref<f32>)

  %accPtr = acc.update_device varPtr(%a : memref<f32>) -> memref<f32>
  acc.update dataOperands(%accPtr : memref<f32>)
  return
}

// CHECK-LABEL: func.func @host_device_ops(
// CHECK-SAME:    %[[A:.*]]: memref<f32>)
// CHECK: %[[DEVPTR_A:.*]] = acc.getdeviceptr varPtr(%[[A]] : memref<f32>)   -> memref<f32>
// CHECK: acc.update_host accPtr(%[[DEVPTR_A]] : memref<f32>) to varPtr(%[[A]] : memref<f32>) {structured = false}
// CHECK: acc.update dataOperands(%[[DEVPTR_A]] : memref<f32>)
// CHECK: %[[DEVPTR_A:.*]] = acc.update_device varPtr(%[[A]] : memref<f32>)   -> memref<f32>
// CHECK: acc.update dataOperands(%[[DEVPTR_A]] : memref<f32>)

// -----

func.func @host_data_ops(%a: !llvm.ptr, %ifCond: i1) -> () {
  %0 = acc.use_device varPtr(%a : !llvm.ptr) -> !llvm.ptr
  acc.host_data dataOperands(%0: !llvm.ptr) {
  }
  acc.host_data dataOperands(%0: !llvm.ptr) {
  } attributes {if_present}
  acc.host_data if(%ifCond) dataOperands(%0: !llvm.ptr) {
  }
  return
}

// CHECK-LABEL: func.func @host_data_ops(
// CHECK-SAME:    %[[A:.*]]: !llvm.ptr, %[[IFCOND:.*]]: i1)
// CHECK: %[[PTR:.*]] = acc.use_device varPtr(%[[A]] : !llvm.ptr) -> !llvm.ptr
// CHECK: acc.host_data dataOperands(%[[PTR]] : !llvm.ptr)
// CHECK: acc.host_data dataOperands(%[[PTR]] : !llvm.ptr) {
// CHECK: } attributes {if_present}
// CHECK: acc.host_data if(%[[IFCOND]]) dataOperands(%[[PTR]] : !llvm.ptr)

// -----

acc.private.recipe @privatization_i32 : !llvm.ptr init {
^bb0(%arg0: !llvm.ptr):
  %c1 = arith.constant 1 : i32
  %c0 = arith.constant 0 : i32
  %0 = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  llvm.store %c0, %0 : i32, !llvm.ptr
  acc.yield %0 : !llvm.ptr
}

// CHECK: acc.private.recipe @privatization_i32 : !llvm.ptr init {
// CHECK: %[[C1:.*]] = arith.constant 1 : i32
// CHECK: %[[C0:.*]] = arith.constant 0 : i32
// CHECK: %[[ALLOCA:.*]] = llvm.alloca %[[C1]] x i32 : (i32) -> !llvm.ptr
// CHECK: llvm.store %[[C0]], %[[ALLOCA]] : i32, !llvm.ptr
// CHECK: acc.yield %[[ALLOCA]] : !llvm.ptr

// -----

func.func private @destroy_struct(!llvm.struct<(i32, i32)>) -> ()

acc.private.recipe @privatization_struct_i32_i64 : !llvm.struct<(i32, i32)> init {
^bb0(%arg0 : !llvm.struct<(i32, i32)>):
  %c1 = arith.constant 1 : i32
  %0 = llvm.mlir.undef : !llvm.struct<(i32, i32)>
  %1 = llvm.insertvalue %c1, %0[0] : !llvm.struct<(i32, i32)>
  %2 = llvm.insertvalue %c1, %1[1] : !llvm.struct<(i32, i32)>
  acc.yield %2 : !llvm.struct<(i32, i32)>
} destroy {
^bb0(%arg0: !llvm.struct<(i32, i32)>):
  func.call @destroy_struct(%arg0) : (!llvm.struct<(i32, i32)>) -> ()
  acc.terminator
}

// CHECK: func.func private @destroy_struct(!llvm.struct<(i32, i32)>)

// CHECK: acc.private.recipe @privatization_struct_i32_i64 : !llvm.struct<(i32, i32)> init {
// CHECK:   %[[C1:.*]] = arith.constant 1 : i32
// CHECK:   %[[UNDEF:.*]] = llvm.mlir.undef : !llvm.struct<(i32, i32)>
// CHECK:   %[[UNDEF1:.*]] = llvm.insertvalue %[[C1]], %[[UNDEF]][0] : !llvm.struct<(i32, i32)> 
// CHECK:   %[[UNDEF2:.*]] = llvm.insertvalue %[[C1]], %[[UNDEF1]][1] : !llvm.struct<(i32, i32)> 
// CHECK:   acc.yield %[[UNDEF2]] : !llvm.struct<(i32, i32)>
// CHECK: } destroy {
// CHECK: ^bb0(%[[ARG0:.*]]: !llvm.struct<(i32, i32)>):
// CHECK:   func.call @destroy_struct(%[[ARG0]]) : (!llvm.struct<(i32, i32)>) -> ()
// CHECK:   acc.terminator

// -----

acc.reduction.recipe @reduction_add_i64 : i64 reduction_operator<add> init {
^bb0(%arg0: i64):
  %0 = arith.constant 0 : i64
  acc.yield %0 : i64
} combiner {
^bb0(%arg0: i64, %arg1: i64):
  %0 = arith.addi %arg0, %arg1 : i64
  acc.yield %0 : i64
}

// CHECK-LABEL: acc.reduction.recipe @reduction_add_i64 : i64 reduction_operator <add> init {
// CHECK:       ^bb0(%{{.*}}: i64):
// CHECK:         %[[C0:.*]] = arith.constant 0 : i64
// CHECK:         acc.yield %[[C0]] : i64
// CHECK:       } combiner {
// CHECK:       ^bb0(%[[ARG0:.*]]: i64, %[[ARG1:.*]]: i64):
// CHECK:         %[[RES:.*]] = arith.addi %[[ARG0]], %[[ARG1]] : i64
// CHECK:         acc.yield %[[RES]] : i64
// CHECK:       }

func.func @acc_reduc_test(%a : i64) -> () {
  acc.parallel reduction(@reduction_add_i64 -> %a : i64) {
    acc.loop reduction(@reduction_add_i64 -> %a : i64) {
      acc.yield
    }
    acc.yield
  }
  return
}

// CHECK-LABEL: func.func @acc_reduc_test(
// CHECK-SAME:    %[[ARG0:.*]]: i64)
// CHECK:         acc.parallel reduction(@reduction_add_i64 -> %[[ARG0]] : i64)
// CHECK:           acc.loop reduction(@reduction_add_i64 -> %[[ARG0]] : i64)

// -----

acc.reduction.recipe @reduction_add_i64 : i64 reduction_operator<add> init {
^bb0(%0: i64):
  %1 = arith.constant 0 : i64
  acc.yield %1 : i64
} combiner {
^bb0(%0: i64, %1: i64):
  %2 = arith.addi %0, %1 : i64
  acc.yield %2 : i64
}

func.func @acc_reduc_test(%a : i64) -> () {
  acc.serial reduction(@reduction_add_i64 -> %a : i64) {
  }
  return
}

// CHECK-LABEL: func.func @acc_reduc_test(
// CHECK-SAME:    %[[ARG0:.*]]: i64)
// CHECK:         acc.serial reduction(@reduction_add_i64 -> %[[ARG0]] : i64)

// -----

func.func @testdeclareop(%a: memref<f32>, %b: memref<f32>, %c: memref<f32>) -> () {
  %0 = acc.copyin varPtr(%a : memref<f32>) -> memref<f32>
  // copyin(zero)
  %1 = acc.copyin varPtr(%b : memref<f32>) -> memref<f32> {dataClause = #acc<data_clause acc_copyin_readonly>}
  // copy
  %2 = acc.copyin varPtr(%c : memref<f32>) -> memref<f32> {dataClause = #acc<data_clause acc_copy>}
  acc.declare_enter dataOperands(%0, %1, %2 : memref<f32>, memref<f32>, memref<f32>)

  %3 = acc.create varPtr(%a : memref<f32>) -> memref<f32>
  // copyout
  %4 = acc.create varPtr(%b : memref<f32>) -> memref<f32> {dataClause = #acc<data_clause acc_copyout>}
  %5 = acc.present varPtr(%c : memref<f32>) -> memref<f32>
  acc.declare_enter dataOperands(%3, %4, %5 : memref<f32>, memref<f32>, memref<f32>)

  %6 = acc.deviceptr varPtr(%a : memref<f32>) -> memref<f32>
  %7 = acc.declare_device_resident varPtr(%b : memref<f32>) -> memref<f32>
  %8 = acc.declare_link varPtr(%c : memref<f32>) -> memref<f32>
  acc.declare_enter dataOperands(%6, %7, %8 : memref<f32>, memref<f32>, memref<f32>)

  acc.declare_exit dataOperands(%7, %8 : memref<f32>, memref<f32>)
  acc.delete accPtr(%7 : memref<f32>) {dataClause = #acc<data_clause acc_declare_device_resident> }
  acc.delete accPtr(%8 : memref<f32>) {dataClause = #acc<data_clause acc_declare_link> }

  acc.declare_exit dataOperands(%3, %4, %5 : memref<f32>, memref<f32>, memref<f32>)
  acc.delete accPtr(%3 : memref<f32>) {dataClause = #acc<data_clause acc_create> }
  acc.copyout accPtr(%4 : memref<f32>) to varPtr(%b : memref<f32>)
  acc.delete accPtr(%5 : memref<f32>) {dataClause = #acc<data_clause acc_present> }

  acc.declare_exit dataOperands(%0, %1, %2 : memref<f32>, memref<f32>, memref<f32>)
  acc.delete accPtr(%0 : memref<f32>) {dataClause = #acc<data_clause acc_copyin> }
  acc.delete accPtr(%1 : memref<f32>) {dataClause = #acc<data_clause acc_copyin_readonly> }
  acc.copyout accPtr(%2 : memref<f32>) to varPtr(%c : memref<f32>) { dataClause = #acc<data_clause acc_copy> }

  return
}

// CHECK-LABEL: func.func @testdeclareop(
// CHECK-SAME: %[[ARGA:.*]]: memref<f32>, %[[ARGB:.*]]: memref<f32>, %[[ARGC:.*]]: memref<f32>)
// CHECK: %[[COPYIN:.*]] = acc.copyin varPtr(%[[ARGA]] : memref<f32>) -> memref<f32>
// CHECK-NEXT: %[[COPYINRO:.*]] = acc.copyin varPtr(%[[ARGB]] : memref<f32>) -> memref<f32> {dataClause = #acc<data_clause acc_copyin_readonly>}
// CHECK-NEXT: %[[COPY:.*]] = acc.copyin varPtr(%[[ARGC]] : memref<f32>) -> memref<f32> {dataClause = #acc<data_clause acc_copy>}
// CHECK-NEXT: acc.declare_enter dataOperands(%[[COPYIN]], %[[COPYINRO]], %[[COPY]] : memref<f32>, memref<f32>, memref<f32>)
// CHECK: %[[CREATE:.*]] = acc.create varPtr(%[[ARGA]] : memref<f32>) -> memref<f32>
// CHECK-NEXT: %[[COPYOUT:.*]] = acc.create varPtr(%[[ARGB]] : memref<f32>) -> memref<f32> {dataClause = #acc<data_clause acc_copyout>}
// CHECK-NEXT: %[[PRESENT:.*]] = acc.present varPtr(%[[ARGC]] : memref<f32>) -> memref<f32>
// CHECK-NEXT: acc.declare_enter dataOperands(%[[CREATE]], %[[COPYOUT]], %[[PRESENT]] : memref<f32>, memref<f32>, memref<f32>)
// CHECK: %[[DEVICEPTR:.*]] = acc.deviceptr varPtr(%[[ARGA]] : memref<f32>) -> memref<f32>
// CHECK-NEXT: %[[DEVICERES:.*]] = acc.declare_device_resident varPtr(%[[ARGB]] : memref<f32>) -> memref<f32>
// CHECK-NEXT: %[[LINK:.*]] = acc.declare_link varPtr(%[[ARGC]] : memref<f32>) -> memref<f32>
// CHECK-NEXT: acc.declare_enter dataOperands(%[[DEVICEPTR]], %[[DEVICERES]], %[[LINK]] : memref<f32>, memref<f32>, memref<f32>)
// CHECK: acc.declare_exit dataOperands(%[[DEVICERES]], %[[LINK]] : memref<f32>, memref<f32>)
// CHECK-NEXT: acc.delete accPtr(%[[DEVICERES]] : memref<f32>) {dataClause = #acc<data_clause acc_declare_device_resident>}
// CHECK-NEXT: acc.delete accPtr(%[[LINK]] : memref<f32>) {dataClause = #acc<data_clause acc_declare_link>}
// CHECK: acc.declare_exit dataOperands(%[[CREATE]], %[[COPYOUT]], %[[PRESENT]] : memref<f32>, memref<f32>, memref<f32>)
// CHECK-NEXT: acc.delete accPtr(%[[CREATE]] : memref<f32>) {dataClause = #acc<data_clause acc_create>}
// CHECK-NEXT: acc.copyout accPtr(%[[COPYOUT]] : memref<f32>) to varPtr(%[[ARGB]] : memref<f32>)
// CHECK-NEXT: acc.delete accPtr(%[[PRESENT]] : memref<f32>) {dataClause = #acc<data_clause acc_present>}
// CHECK: acc.declare_exit dataOperands(%[[COPYIN]], %[[COPYINRO]], %[[COPY]] : memref<f32>, memref<f32>, memref<f32>)
// CHECK-NEXT: acc.delete accPtr(%[[COPYIN]] : memref<f32>) {dataClause = #acc<data_clause acc_copyin>}
// CHECK-NEXT: acc.delete accPtr(%[[COPYINRO]] : memref<f32>) {dataClause = #acc<data_clause acc_copyin_readonly>}
// CHECK-NEXT: acc.copyout accPtr(%[[COPY]] : memref<f32>) to varPtr(%[[ARGC]] : memref<f32>) {dataClause = #acc<data_clause acc_copy>}

// -----

llvm.mlir.global external @globalvar() { acc.declare = #acc.declare<dataClause = acc_create> } : i32 {
  %0 = llvm.mlir.constant(0 : i32) : i32
  llvm.return %0 : i32
}

acc.global_ctor @acc_constructor {
  %0 = llvm.mlir.addressof @globalvar { acc.declare = #acc.declare<dataClause = acc_create> } : !llvm.ptr
  %1 = acc.create varPtr(%0 : !llvm.ptr) -> !llvm.ptr
  acc.declare_enter dataOperands(%1 : !llvm.ptr)
  acc.terminator
}

acc.global_dtor @acc_destructor {
  %0 = llvm.mlir.addressof @globalvar { acc.declare = #acc.declare<dataClause = acc_create> } : !llvm.ptr
  %1 = acc.getdeviceptr varPtr(%0 : !llvm.ptr) -> !llvm.ptr {dataClause = #acc<data_clause acc_create>}
  acc.declare_exit dataOperands(%1 : !llvm.ptr)
  acc.delete accPtr(%1 : !llvm.ptr)
  acc.terminator
}

// CHECK-LABEL: acc.global_ctor @acc_constructor
// CHECK: %[[ADDR:.*]] = llvm.mlir.addressof @globalvar {acc.declare = #acc.declare<dataClause = acc_create>} : !llvm.ptr
// CHECK-NEXT: %[[CREATE:.*]] = acc.create varPtr(%[[ADDR]] : !llvm.ptr) -> !llvm.ptr
// CHECK-NEXT: acc.declare_enter dataOperands(%[[CREATE]] : !llvm.ptr)
// CHECK: acc.global_dtor @acc_destructor
// CHECK: %[[ADDR:.*]] = llvm.mlir.addressof @globalvar {acc.declare = #acc.declare<dataClause = acc_create>} : !llvm.ptr
// CHECK-NEXT: %[[DELETE:.*]] = acc.getdeviceptr varPtr(%[[ADDR]] : !llvm.ptr) -> !llvm.ptr {dataClause = #acc<data_clause acc_create>}
// CHECK-NEXT: acc.declare_exit dataOperands(%[[DELETE]] : !llvm.ptr)
// CHECK-NEXT: acc.delete accPtr(%[[DELETE]] : !llvm.ptr)

// -----

func.func @acc_func(%a : i64) -> () attributes {acc.routine_info = #acc.routine_info<[@acc_func_rout1,@acc_func_rout2,@acc_func_rout3,
    @acc_func_rout4,@acc_func_rout5,@acc_func_rout6,@acc_func_rout7,@acc_func_rout8,@acc_func_rout9]>} {
  return
}

acc.routine @acc_func_rout1 func(@acc_func)
acc.routine @acc_func_rout2 func(@acc_func) bind("acc_func_gpu")
acc.routine @acc_func_rout3 func(@acc_func) bind("acc_func_gpu_gang") gang
acc.routine @acc_func_rout4 func(@acc_func) bind("acc_func_gpu_vector") vector
acc.routine @acc_func_rout5 func(@acc_func) bind("acc_func_gpu_worker") worker
acc.routine @acc_func_rout6 func(@acc_func) bind("acc_func_gpu_seq") seq
acc.routine @acc_func_rout7 func(@acc_func) bind("acc_func_gpu_imp_gang") implicit gang
acc.routine @acc_func_rout8 func(@acc_func) bind("acc_func_gpu_vector_nohost") vector nohost
acc.routine @acc_func_rout9 func(@acc_func) bind("acc_func_gpu_gang_dim1") gang(dim = 1 : i32)

// CHECK-LABEL: func.func @acc_func(
// CHECK: attributes {acc.routine_info = #acc.routine_info<[@acc_func_rout1, @acc_func_rout2, @acc_func_rout3,
// CHECK: @acc_func_rout4, @acc_func_rout5, @acc_func_rout6, @acc_func_rout7, @acc_func_rout8, @acc_func_rout9]>}
// CHECK: acc.routine @acc_func_rout1 func(@acc_func)
// CHECK: acc.routine @acc_func_rout2 func(@acc_func) bind("acc_func_gpu")
// CHECK: acc.routine @acc_func_rout3 func(@acc_func) bind("acc_func_gpu_gang") gang
// CHECK: acc.routine @acc_func_rout4 func(@acc_func) bind("acc_func_gpu_vector") vector
// CHECK: acc.routine @acc_func_rout5 func(@acc_func) bind("acc_func_gpu_worker") worker
// CHECK: acc.routine @acc_func_rout6 func(@acc_func) bind("acc_func_gpu_seq") seq
// CHECK: acc.routine @acc_func_rout7 func(@acc_func) bind("acc_func_gpu_imp_gang") gang implicit
// CHECK: acc.routine @acc_func_rout8 func(@acc_func) bind("acc_func_gpu_vector_nohost") vector nohost
// CHECK: acc.routine @acc_func_rout9 func(@acc_func) bind("acc_func_gpu_gang_dim1") gang(dim = 1 : i32)

// -----

func.func @acc_func() -> () {
  "test.openacc_dummy_op"() {acc.declare_action = #acc.declare_action<postAlloc = @_QMacc_declareFacc_declare_allocateEa_acc_declare_update_desc_post_alloc>} : () -> ()
  return
}

// CHECK-LABEL: func.func @acc_func
// CHECK: "test.openacc_dummy_op"() {acc.declare_action = #acc.declare_action<postAlloc = @_QMacc_declareFacc_declare_allocateEa_acc_declare_update_desc_post_alloc>}

// -----

func.func @compute3(%a: memref<10x10xf32>, %b: memref<10x10xf32>, %c: memref<10xf32>, %d: memref<10xf32>) {
  %lb = arith.constant 0 : index
  %st = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %numGangs = arith.constant 10 : i64
  %numWorkers = arith.constant 10 : i64

  %c20 = arith.constant 20 : i32
  %alloc = llvm.alloca %c20 x i32 { acc.declare = #acc.declare<dataClause = acc_create, implicit = true> } : (i32) -> !llvm.ptr
  %createlocal = acc.create varPtr(%alloc : !llvm.ptr) -> !llvm.ptr {implicit = true}

  %pa = acc.present varPtr(%a : memref<10x10xf32>) -> memref<10x10xf32>
  %pb = acc.present varPtr(%b : memref<10x10xf32>) -> memref<10x10xf32>
  %pc = acc.present varPtr(%c : memref<10xf32>) -> memref<10xf32>
  %pd = acc.present varPtr(%d : memref<10xf32>) -> memref<10xf32>
  acc.declare dataOperands(%pa, %pb, %pc, %pd, %createlocal: memref<10x10xf32>, memref<10x10xf32>, memref<10xf32>, memref<10xf32>, !llvm.ptr) {
  }

  return
}

// CHECK-LABEL: func.func @compute3
// CHECK: acc.declare dataOperands(

// -----

%i64Value = arith.constant 1 : i64
%i32Value = arith.constant 1 : i32
%i32Value2 = arith.constant 2 : i32
%idxValue = arith.constant 1 : index
%ifCond = arith.constant true
acc.set attributes {device_type = #acc.device_type<nvidia>}
acc.set device_num(%i64Value : i64)
acc.set device_num(%i32Value : i32)
acc.set device_num(%idxValue : index)
acc.set device_num(%idxValue : index) if(%ifCond)
acc.set default_async(%i32Value : i32)

// CHECK: [[I64VALUE:%.*]] = arith.constant 1 : i64
// CHECK: [[I32VALUE:%.*]] = arith.constant 1 : i32
// CHECK: [[I32VALUE2:%.*]] = arith.constant 2 : i32
// CHECK: [[IDXVALUE:%.*]] = arith.constant 1 : index
// CHECK: [[IFCOND:%.*]] = arith.constant true
// CHECK: acc.set attributes {device_type = #acc.device_type<nvidia>}
// CHECK: acc.set device_num([[I64VALUE]] : i64)
// CHECK: acc.set device_num([[I32VALUE]] : i32)
// CHECK: acc.set device_num([[IDXVALUE]] : index)
// CHECK: acc.set device_num([[IDXVALUE]] : index) if([[IFCOND]])
// CHECK: acc.set default_async([[I32VALUE]] : i32)

// -----

// CHECK-LABEL: func.func @acc_atomic_read
// CHECK-SAME: (%[[v:.*]]: memref<i32>, %[[x:.*]]: memref<i32>)
func.func @acc_atomic_read(%v: memref<i32>, %x: memref<i32>) {
  // CHECK: acc.atomic.read %[[v]] = %[[x]] : memref<i32>, i32
  acc.atomic.read %v = %x : memref<i32>, i32
  return
}

// -----

// CHECK-LABEL: func.func @acc_atomic_write
// CHECK-SAME: (%[[ADDR:.*]]: memref<i32>, %[[VAL:.*]]: i32)
func.func @acc_atomic_write(%addr : memref<i32>, %val : i32) {
  // CHECK: acc.atomic.write %[[ADDR]] = %[[VAL]] : memref<i32>, i32
  acc.atomic.write %addr = %val : memref<i32>, i32
  return
}

// -----

// CHECK-LABEL: func.func @acc_atomic_update
// CHECK-SAME: (%[[X:.*]]: memref<i32>, %[[EXPR:.*]]: i32, %[[XBOOL:.*]]: memref<i1>, %[[EXPRBOOL:.*]]: i1)
func.func @acc_atomic_update(%x : memref<i32>, %expr : i32, %xBool : memref<i1>, %exprBool : i1) {
  // CHECK: acc.atomic.update %[[X]] : memref<i32>
  // CHECK-NEXT: (%[[XVAL:.*]]: i32):
  // CHECK-NEXT:   %[[NEWVAL:.*]] = llvm.add %[[XVAL]], %[[EXPR]] : i32
  // CHECK-NEXT:   acc.yield %[[NEWVAL]] : i32
  acc.atomic.update %x : memref<i32> {
  ^bb0(%xval: i32):
    %newval = llvm.add %xval, %expr : i32
    acc.yield %newval : i32
  }
  // CHECK: acc.atomic.update %[[XBOOL]] : memref<i1>
  // CHECK-NEXT: (%[[XVAL:.*]]: i1):
  // CHECK-NEXT:   %[[NEWVAL:.*]] = llvm.and %[[XVAL]], %[[EXPRBOOL]] : i1
  // CHECK-NEXT:   acc.yield %[[NEWVAL]] : i1
  acc.atomic.update %xBool : memref<i1> {
  ^bb0(%xval: i1):
    %newval = llvm.and %xval, %exprBool : i1
    acc.yield %newval : i1
  }
  // CHECK: acc.atomic.update %[[X]] : memref<i32>
  // CHECK-NEXT: (%[[XVAL:.*]]: i32):
  // CHECK-NEXT:   %[[NEWVAL:.*]] = llvm.shl %[[XVAL]], %[[EXPR]] : i32
  // CHECK-NEXT:   acc.yield %[[NEWVAL]] : i32
  // CHECK-NEXT: }
  acc.atomic.update %x : memref<i32> {
  ^bb0(%xval: i32):
    %newval = llvm.shl %xval, %expr : i32
    acc.yield %newval : i32
  }
  // CHECK: acc.atomic.update %[[X]] : memref<i32>
  // CHECK-NEXT: (%[[XVAL:.*]]: i32):
  // CHECK-NEXT:   %[[NEWVAL:.*]] = llvm.intr.smax(%[[XVAL]], %[[EXPR]]) : (i32, i32) -> i32
  // CHECK-NEXT:   acc.yield %[[NEWVAL]] : i32
  // CHECK-NEXT: }
  acc.atomic.update %x : memref<i32> {
  ^bb0(%xval: i32):
    %newval = llvm.intr.smax(%xval, %expr) : (i32, i32) -> i32
    acc.yield %newval : i32
  }

  // CHECK: acc.atomic.update %[[XBOOL]] : memref<i1>
  // CHECK-NEXT: (%[[XVAL:.*]]: i1):
  // CHECK-NEXT:   %[[NEWVAL:.*]] = llvm.icmp "eq" %[[XVAL]], %[[EXPRBOOL]] : i1
  // CHECK-NEXT:   acc.yield %[[NEWVAL]] : i1
  // }
  acc.atomic.update %xBool : memref<i1> {
  ^bb0(%xval: i1):
    %newval = llvm.icmp "eq" %xval, %exprBool : i1
    acc.yield %newval : i1
  }

  // CHECK: acc.atomic.update %[[X]] : memref<i32> {
  // CHECK-NEXT: (%[[XVAL:.*]]: i32):
  // CHECK-NEXT:   acc.yield %[[XVAL]] : i32
  // CHECK-NEXT: }
  acc.atomic.update %x : memref<i32> {
  ^bb0(%xval:i32):
    acc.yield %xval : i32
  }

  // CHECK: acc.atomic.update %[[X]] : memref<i32> {
  // CHECK-NEXT: (%[[XVAL:.*]]: i32):
  // CHECK-NEXT:   acc.yield %{{.+}} : i32
  // CHECK-NEXT: }
  %const = arith.constant 42 : i32
  acc.atomic.update %x : memref<i32> {
  ^bb0(%xval:i32):
    acc.yield %const : i32
  }

  return
}

// -----

// CHECK-LABEL: func.func @acc_atomic_capture
// CHECK-SAME: (%[[v:.*]]: memref<i32>, %[[x:.*]]: memref<i32>, %[[expr:.*]]: i32)
func.func @acc_atomic_capture(%v: memref<i32>, %x: memref<i32>, %expr: i32) {
  // CHECK: acc.atomic.capture {
  // CHECK-NEXT: acc.atomic.update %[[x]] : memref<i32>
  // CHECK-NEXT: (%[[xval:.*]]: i32):
  // CHECK-NEXT:   %[[newval:.*]] = llvm.add %[[xval]], %[[expr]] : i32
  // CHECK-NEXT:   acc.yield %[[newval]] : i32
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.atomic.read %[[v]] = %[[x]] : memref<i32>, i32
  // CHECK-NEXT: }
  acc.atomic.capture {
    acc.atomic.update %x : memref<i32> {
    ^bb0(%xval: i32):
      %newval = llvm.add %xval, %expr : i32
      acc.yield %newval : i32
    }
    acc.atomic.read %v = %x : memref<i32>, i32
  }
  // CHECK: acc.atomic.capture {
  // CHECK-NEXT: acc.atomic.read %[[v]] = %[[x]] : memref<i32>, i32
  // CHECK-NEXT: acc.atomic.update %[[x]] : memref<i32>
  // CHECK-NEXT: (%[[xval:.*]]: i32):
  // CHECK-NEXT:   %[[newval:.*]] = llvm.add %[[xval]], %[[expr]] : i32
  // CHECK-NEXT:   acc.yield %[[newval]] : i32
  // CHECK-NEXT: }
  // CHECK-NEXT: }
  acc.atomic.capture {
    acc.atomic.read %v = %x : memref<i32>, i32
    acc.atomic.update %x : memref<i32> {
    ^bb0(%xval: i32):
      %newval = llvm.add %xval, %expr : i32
      acc.yield %newval : i32
    }
  }
  // CHECK: acc.atomic.capture {
  // CHECK-NEXT: acc.atomic.read %[[v]] = %[[x]] : memref<i32>, i32
  // CHECK-NEXT: acc.atomic.write %[[x]] = %[[expr]] : memref<i32>, i32
  // CHECK-NEXT: }
  acc.atomic.capture {
    acc.atomic.read %v = %x : memref<i32>, i32
    acc.atomic.write %x = %expr : memref<i32>, i32
  }

  return
}
