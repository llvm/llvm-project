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

func.func @compute3(%a: memref<10x10xf32>, %b: memref<10x10xf32>, %c: memref<10xf32>, %d: memref<10xf32>) -> memref<10xf32> {
  %lb = arith.constant 0 : index
  %st = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %numGangs = arith.constant 10 : i64
  %numWorkers = arith.constant 10 : i64

  acc.data present(%a, %b, %c, %d: memref<10x10xf32>, memref<10x10xf32>, memref<10xf32>, memref<10xf32>) {
    acc.parallel num_gangs(%numGangs: i64) num_workers(%numWorkers: i64) private(%c : memref<10xf32>) {
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
// CHECK-NEXT:   acc.data present(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : memref<10x10xf32>, memref<10x10xf32>, memref<10xf32>, memref<10xf32>) {
// CHECK-NEXT:     acc.parallel num_gangs([[NUMGANG]] : i64) num_workers([[NUMWORKERS]] : i64) private([[ARG2]] : memref<10xf32>) {
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

func.func @testloopop() -> () {
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
  acc.parallel copyin(%a, %b : memref<10xf32>, memref<10xf32>) {
  }
  acc.parallel copyin_readonly(%a, %b : memref<10xf32>, memref<10xf32>) {
  }
  acc.parallel copyin(%a: memref<10xf32>) copyout_zero(%b, %c : memref<10xf32>, memref<10x10xf32>) {
  }
  acc.parallel copyout(%b, %c : memref<10xf32>, memref<10x10xf32>) create(%a: memref<10xf32>) {
  }
  acc.parallel copyout_zero(%b, %c : memref<10xf32>, memref<10x10xf32>) create_zero(%a: memref<10xf32>) {
  }
  acc.parallel no_create(%a: memref<10xf32>) present(%b, %c : memref<10xf32>, memref<10x10xf32>) {
  }
  acc.parallel deviceptr(%a: memref<10xf32>) attach(%b, %c : memref<10xf32>, memref<10x10xf32>) {
  }
  acc.parallel private(%a, %c : memref<10xf32>, memref<10x10xf32>) firstprivate(%b: memref<10xf32>) {
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
// CHECK:      acc.parallel copyin([[ARGA]], [[ARGB]] : memref<10xf32>, memref<10xf32>) {
// CHECK-NEXT: }
// CHECK:      acc.parallel copyin_readonly([[ARGA]], [[ARGB]] : memref<10xf32>, memref<10xf32>) {
// CHECK-NEXT: }
// CHECK:      acc.parallel copyin([[ARGA]] : memref<10xf32>) copyout_zero([[ARGB]], [[ARGC]] : memref<10xf32>, memref<10x10xf32>) {
// CHECK-NEXT: }
// CHECK:      acc.parallel copyout([[ARGB]], [[ARGC]] : memref<10xf32>, memref<10x10xf32>) create([[ARGA]] : memref<10xf32>) {
// CHECK-NEXT: }
// CHECK:      acc.parallel copyout_zero([[ARGB]], [[ARGC]] : memref<10xf32>, memref<10x10xf32>) create_zero([[ARGA]] : memref<10xf32>) {
// CHECK-NEXT: }
// CHECK:      acc.parallel no_create([[ARGA]] : memref<10xf32>) present([[ARGB]], [[ARGC]] : memref<10xf32>, memref<10x10xf32>) {
// CHECK-NEXT: }
// CHECK:      acc.parallel attach([[ARGB]], [[ARGC]] : memref<10xf32>, memref<10x10xf32>) deviceptr([[ARGA]] : memref<10xf32>) {
// CHECK-NEXT: }
// CHECK:      acc.parallel firstprivate([[ARGB]] : memref<10xf32>) private([[ARGA]], [[ARGC]] : memref<10xf32>, memref<10x10xf32>) {
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

// -----

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
  acc.serial copyin(%a, %b : memref<10xf32>, memref<10xf32>) {
  }
  acc.serial copyin_readonly(%a, %b : memref<10xf32>, memref<10xf32>) {
  }
  acc.serial copyin(%a: memref<10xf32>) copyout_zero(%b, %c : memref<10xf32>, memref<10x10xf32>) {
  }
  acc.serial copyout(%b, %c : memref<10xf32>, memref<10x10xf32>) create(%a: memref<10xf32>) {
  }
  acc.serial copyout_zero(%b, %c : memref<10xf32>, memref<10x10xf32>) create_zero(%a: memref<10xf32>) {
  }
  acc.serial no_create(%a: memref<10xf32>) present(%b, %c : memref<10xf32>, memref<10x10xf32>) {
  }
  acc.serial deviceptr(%a: memref<10xf32>) attach(%b, %c : memref<10xf32>, memref<10x10xf32>) {
  }
  acc.serial private(%a, %c : memref<10xf32>, memref<10x10xf32>) firstprivate(%b: memref<10xf32>) {
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
// CHECK:      acc.serial copyin([[ARGA]], [[ARGB]] : memref<10xf32>, memref<10xf32>) {
// CHECK-NEXT: }
// CHECK:      acc.serial copyin_readonly([[ARGA]], [[ARGB]] : memref<10xf32>, memref<10xf32>) {
// CHECK-NEXT: }
// CHECK:      acc.serial copyin([[ARGA]] : memref<10xf32>) copyout_zero([[ARGB]], [[ARGC]] : memref<10xf32>, memref<10x10xf32>) {
// CHECK-NEXT: }
// CHECK:      acc.serial copyout([[ARGB]], [[ARGC]] : memref<10xf32>, memref<10x10xf32>) create([[ARGA]] : memref<10xf32>) {
// CHECK-NEXT: }
// CHECK:      acc.serial copyout_zero([[ARGB]], [[ARGC]] : memref<10xf32>, memref<10x10xf32>) create_zero([[ARGA]] : memref<10xf32>) {
// CHECK-NEXT: }
// CHECK:      acc.serial no_create([[ARGA]] : memref<10xf32>) present([[ARGB]], [[ARGC]] : memref<10xf32>, memref<10x10xf32>) {
// CHECK-NEXT: }
// CHECK:      acc.serial attach([[ARGB]], [[ARGC]] : memref<10xf32>, memref<10x10xf32>) deviceptr([[ARGA]] : memref<10xf32>) {
// CHECK-NEXT: }
// CHECK:      acc.serial firstprivate([[ARGB]] : memref<10xf32>) private([[ARGA]], [[ARGC]] : memref<10xf32>, memref<10x10xf32>) {
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
  acc.kernels copyin(%a, %b : memref<10xf32>, memref<10xf32>) {
  }
  acc.kernels copyin_readonly(%a, %b : memref<10xf32>, memref<10xf32>) {
  }
  acc.kernels copyin(%a: memref<10xf32>) copyout_zero(%b, %c : memref<10xf32>, memref<10x10xf32>) {
  }
  acc.kernels copyout(%b, %c : memref<10xf32>, memref<10x10xf32>) create(%a: memref<10xf32>) {
  }
  acc.kernels copyout_zero(%b, %c : memref<10xf32>, memref<10x10xf32>) create_zero(%a: memref<10xf32>) {
  }
  acc.kernels no_create(%a: memref<10xf32>) present(%b, %c : memref<10xf32>, memref<10x10xf32>) {
  }
  acc.kernels deviceptr(%a: memref<10xf32>) attach(%b, %c : memref<10xf32>, memref<10x10xf32>) {
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
// CHECK:      acc.kernels copyin([[ARGA]], [[ARGB]] : memref<10xf32>, memref<10xf32>) {
// CHECK-NEXT: }
// CHECK:      acc.kernels copyin_readonly([[ARGA]], [[ARGB]] : memref<10xf32>, memref<10xf32>) {
// CHECK-NEXT: }
// CHECK:      acc.kernels copyin([[ARGA]] : memref<10xf32>) copyout_zero([[ARGB]], [[ARGC]] : memref<10xf32>, memref<10x10xf32>) {
// CHECK-NEXT: }
// CHECK:      acc.kernels copyout([[ARGB]], [[ARGC]] : memref<10xf32>, memref<10x10xf32>) create([[ARGA]] : memref<10xf32>) {
// CHECK-NEXT: }
// CHECK:      acc.kernels copyout_zero([[ARGB]], [[ARGC]] : memref<10xf32>, memref<10x10xf32>) create_zero([[ARGA]] : memref<10xf32>) {
// CHECK-NEXT: }
// CHECK:      acc.kernels no_create([[ARGA]] : memref<10xf32>) present([[ARGB]], [[ARGC]] : memref<10xf32>, memref<10x10xf32>) {
// CHECK-NEXT: }
// CHECK:      acc.kernels attach([[ARGB]], [[ARGC]] : memref<10xf32>, memref<10x10xf32>) deviceptr([[ARGA]] : memref<10xf32>) {
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

func.func @testdataop(%a: memref<10xf32>, %b: memref<10xf32>, %c: memref<10x10xf32>) -> () {
  %ifCond = arith.constant true
  acc.data if(%ifCond) present(%a : memref<10xf32>) {
  }
  acc.data present(%a : memref<10xf32>) if(%ifCond) {
  }
  acc.data present(%a, %b, %c : memref<10xf32>, memref<10xf32>, memref<10x10xf32>) {
  }
  acc.data copy(%a, %b, %c : memref<10xf32>, memref<10xf32>, memref<10x10xf32>) {
  }
  acc.data copyin(%a, %b, %c : memref<10xf32>, memref<10xf32>, memref<10x10xf32>) {
  }
  acc.data copyin_readonly(%a, %b, %c : memref<10xf32>, memref<10xf32>, memref<10x10xf32>) {
  }
  acc.data copyout(%a, %b, %c : memref<10xf32>, memref<10xf32>, memref<10x10xf32>) {
  }
  acc.data copyout_zero(%a, %b, %c : memref<10xf32>, memref<10xf32>, memref<10x10xf32>) {
  }
  acc.data create(%a, %b, %c : memref<10xf32>, memref<10xf32>, memref<10x10xf32>) {
  }
  acc.data create_zero(%a, %b, %c : memref<10xf32>, memref<10xf32>, memref<10x10xf32>) {
  }
  acc.data no_create(%a, %b, %c : memref<10xf32>, memref<10xf32>, memref<10x10xf32>) {
  }
  acc.data deviceptr(%a, %b, %c : memref<10xf32>, memref<10xf32>, memref<10x10xf32>) {
  }
  acc.data attach(%a, %b, %c : memref<10xf32>, memref<10xf32>, memref<10x10xf32>) {
  }
  acc.data copyin(%b: memref<10xf32>) copyout(%c: memref<10x10xf32>) present(%a: memref<10xf32>) {
  }
  acc.data present(%a : memref<10xf32>) {
  } attributes { defaultAttr = #acc<defaultvalue none> }
  acc.data present(%a : memref<10xf32>) {
  } attributes { defaultAttr = #acc<defaultvalue present> }
  acc.data {
  } attributes { defaultAttr = #acc<defaultvalue none> }
  return
}

// CHECK:      func @testdataop([[ARGA:%.*]]: memref<10xf32>, [[ARGB:%.*]]: memref<10xf32>, [[ARGC:%.*]]: memref<10x10xf32>) {
// CHECK:      [[IFCOND1:%.*]] = arith.constant true
// CHECK:      acc.data if([[IFCOND1]]) present([[ARGA]] : memref<10xf32>) {
// CHECK-NEXT: }
// CHECK:      acc.data if([[IFCOND1]]) present([[ARGA]] : memref<10xf32>) {
// CHECK-NEXT: }
// CHECK:      acc.data present([[ARGA]], [[ARGB]], [[ARGC]] : memref<10xf32>, memref<10xf32>, memref<10x10xf32>) {
// CHECK-NEXT: }
// CHECK:      acc.data copy([[ARGA]], [[ARGB]], [[ARGC]] : memref<10xf32>, memref<10xf32>, memref<10x10xf32>) {
// CHECK-NEXT: }
// CHECK:      acc.data copyin([[ARGA]], [[ARGB]], [[ARGC]] : memref<10xf32>, memref<10xf32>, memref<10x10xf32>) {
// CHECK-NEXT: }
// CHECK:      acc.data copyin_readonly([[ARGA]], [[ARGB]], [[ARGC]] : memref<10xf32>, memref<10xf32>, memref<10x10xf32>) {
// CHECK-NEXT: }
// CHECK:      acc.data copyout([[ARGA]], [[ARGB]], [[ARGC]] : memref<10xf32>, memref<10xf32>, memref<10x10xf32>) {
// CHECK-NEXT: }
// CHECK:      acc.data copyout_zero([[ARGA]], [[ARGB]], [[ARGC]] : memref<10xf32>, memref<10xf32>, memref<10x10xf32>) {
// CHECK-NEXT: }
// CHECK:      acc.data create([[ARGA]], [[ARGB]], [[ARGC]] : memref<10xf32>, memref<10xf32>, memref<10x10xf32>) {
// CHECK-NEXT: }
// CHECK:      acc.data create_zero([[ARGA]], [[ARGB]], [[ARGC]] : memref<10xf32>, memref<10xf32>, memref<10x10xf32>) {
// CHECK-NEXT: }
// CHECK:      acc.data no_create([[ARGA]], [[ARGB]], [[ARGC]] : memref<10xf32>, memref<10xf32>, memref<10x10xf32>) {
// CHECK-NEXT: }
// CHECK:      acc.data deviceptr([[ARGA]], [[ARGB]], [[ARGC]] : memref<10xf32>, memref<10xf32>, memref<10x10xf32>) {
// CHECK-NEXT: }
// CHECK:      acc.data attach([[ARGA]], [[ARGB]], [[ARGC]] : memref<10xf32>, memref<10xf32>, memref<10x10xf32>) {
// CHECK-NEXT: }
// CHECK:      acc.data copyin([[ARGB]] : memref<10xf32>) copyout([[ARGC]] : memref<10x10xf32>) present([[ARGA]] : memref<10xf32>) {
// CHECK-NEXT: }
// CHECK:      acc.data present([[ARGA]] : memref<10xf32>) {
// CHECK-NEXT: } attributes {defaultAttr = #acc<defaultvalue none>}
// CHECK:      acc.data present([[ARGA]] : memref<10xf32>) {
// CHECK-NEXT: } attributes {defaultAttr = #acc<defaultvalue present>}
// CHECK:      acc.data {
// CHECK-NEXT: } attributes {defaultAttr = #acc<defaultvalue none>}

// -----

func.func @testupdateop(%a: memref<10xf32>, %b: memref<10xf32>, %c: memref<10x10xf32>) -> () {
  %i64Value = arith.constant 1 : i64
  %i32Value = arith.constant 1 : i32
  %idxValue = arith.constant 1 : index
  %ifCond = arith.constant true
  acc.update async(%i64Value: i64) host(%a: memref<10xf32>)
  acc.update async(%i32Value: i32) host(%a: memref<10xf32>)
  acc.update async(%i32Value: i32) host(%a: memref<10xf32>)
  acc.update async(%idxValue: index) host(%a: memref<10xf32>)
  acc.update wait_devnum(%i64Value: i64) wait(%i32Value, %idxValue : i32, index) host(%a: memref<10xf32>)
  acc.update if(%ifCond) host(%a: memref<10xf32>)
  acc.update device_type(%i32Value : i32) host(%a: memref<10xf32>)
  acc.update host(%a: memref<10xf32>) device(%b, %c : memref<10xf32>, memref<10x10xf32>)
  acc.update host(%a: memref<10xf32>) device(%b, %c : memref<10xf32>, memref<10x10xf32>) attributes {async}
  acc.update host(%a: memref<10xf32>) device(%b, %c : memref<10xf32>, memref<10x10xf32>) attributes {wait}
  acc.update host(%a: memref<10xf32>) device(%b, %c : memref<10xf32>, memref<10x10xf32>) attributes {ifPresent}
  return
}

// CHECK: func @testupdateop([[ARGA:%.*]]: memref<10xf32>, [[ARGB:%.*]]: memref<10xf32>, [[ARGC:%.*]]: memref<10x10xf32>) {
// CHECK:   [[I64VALUE:%.*]] = arith.constant 1 : i64
// CHECK:   [[I32VALUE:%.*]] = arith.constant 1 : i32
// CHECK:   [[IDXVALUE:%.*]] = arith.constant 1 : index
// CHECK:   [[IFCOND:%.*]] = arith.constant true
// CHECK:   acc.update async([[I64VALUE]] : i64) host([[ARGA]] : memref<10xf32>)
// CHECK:   acc.update async([[I32VALUE]] : i32) host([[ARGA]] : memref<10xf32>)
// CHECK:   acc.update async([[I32VALUE]] : i32) host([[ARGA]] : memref<10xf32>)
// CHECK:   acc.update async([[IDXVALUE]] : index) host([[ARGA]] : memref<10xf32>)
// CHECK:   acc.update wait_devnum([[I64VALUE]] : i64) wait([[I32VALUE]], [[IDXVALUE]] : i32, index) host([[ARGA]] : memref<10xf32>)
// CHECK:   acc.update if([[IFCOND]]) host([[ARGA]] : memref<10xf32>)
// CHECK:   acc.update device_type([[I32VALUE]] : i32) host([[ARGA]] : memref<10xf32>)
// CHECK:   acc.update host([[ARGA]] : memref<10xf32>) device([[ARGB]], [[ARGC]] : memref<10xf32>, memref<10x10xf32>)
// CHECK:   acc.update host([[ARGA]] : memref<10xf32>) device([[ARGB]], [[ARGC]] : memref<10xf32>, memref<10x10xf32>) attributes {async}
// CHECK:   acc.update host([[ARGA]] : memref<10xf32>) device([[ARGB]], [[ARGC]] : memref<10xf32>, memref<10x10xf32>) attributes {wait}
// CHECK:   acc.update host([[ARGA]] : memref<10xf32>) device([[ARGB]], [[ARGC]] : memref<10xf32>, memref<10x10xf32>) attributes {ifPresent}

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
acc.init device_type(%i32Value : i32)
acc.init device_type(%i32Value, %i32Value2 : i32, i32)
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
// CHECK: acc.init device_type([[I32VALUE]] : i32)
// CHECK: acc.init device_type([[I32VALUE]], [[I32VALUE2]] : i32, i32)
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
acc.shutdown device_type(%i32Value : i32)
acc.shutdown device_type(%i32Value, %i32Value2 : i32, i32)
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
// CHECK: acc.shutdown device_type([[I32VALUE]] : i32)
// CHECK: acc.shutdown device_type([[I32VALUE]], [[I32VALUE2]] : i32, i32)
// CHECK: acc.shutdown device_num([[I64VALUE]] : i64)
// CHECK: acc.shutdown device_num([[I32VALUE]] : i32)
// CHECK: acc.shutdown device_num([[IDXVALUE]] : index)
// CHECK: acc.shutdown if([[IFCOND]])
// CHECK: acc.shutdown device_num([[IDXVALUE]] : index) if([[IFCOND]])
// CHECK: acc.shutdown device_num([[IDXVALUE]] : index) if([[IFCOND]])

// -----

func.func @testexitdataop(%a: memref<10xf32>, %b: memref<10xf32>, %c: memref<10x10xf32>) -> () {
  %ifCond = arith.constant true
  %i64Value = arith.constant 1 : i64
  %i32Value = arith.constant 1 : i32
  %idxValue = arith.constant 1 : index

  acc.exit_data copyout(%a : memref<10xf32>)
  acc.exit_data delete(%a : memref<10xf32>)
  acc.exit_data delete(%a : memref<10xf32>) attributes {async,finalize}
  acc.exit_data detach(%a : memref<10xf32>)
  acc.exit_data copyout(%a : memref<10xf32>) attributes {async}
  acc.exit_data delete(%a : memref<10xf32>) attributes {wait}
  acc.exit_data async(%i64Value : i64) copyout(%a : memref<10xf32>)
  acc.exit_data copyout(%a : memref<10xf32>) async(%i64Value : i64)
  acc.exit_data if(%ifCond) copyout(%a : memref<10xf32>)
  acc.exit_data wait_devnum(%i64Value: i64) wait(%i32Value, %idxValue : i32, index) copyout(%a : memref<10xf32>)

  return
}

// CHECK: func @testexitdataop([[ARGA:%.*]]: memref<10xf32>, [[ARGB:%.*]]: memref<10xf32>, [[ARGC:%.*]]: memref<10x10xf32>) {
// CHECK: [[IFCOND1:%.*]] = arith.constant true
// CHECK: [[I64VALUE:%.*]] = arith.constant 1 : i64
// CHECK: [[I32VALUE:%.*]] = arith.constant 1 : i32
// CHECK: [[IDXVALUE:%.*]] = arith.constant 1 : index
// CHECK: acc.exit_data copyout([[ARGA]] : memref<10xf32>)
// CHECK: acc.exit_data delete([[ARGA]] : memref<10xf32>)
// CHECK: acc.exit_data delete([[ARGA]] : memref<10xf32>) attributes {async, finalize}
// CHECK: acc.exit_data detach([[ARGA]] : memref<10xf32>)
// CHECK: acc.exit_data copyout([[ARGA]] : memref<10xf32>) attributes {async}
// CHECK: acc.exit_data delete([[ARGA]] : memref<10xf32>) attributes {wait}
// CHECK: acc.exit_data async([[I64VALUE]] : i64) copyout([[ARGA]] : memref<10xf32>)
// CHECK: acc.exit_data async([[I64VALUE]] : i64) copyout([[ARGA]] : memref<10xf32>)
// CHECK: acc.exit_data if([[IFCOND]]) copyout([[ARGA]] : memref<10xf32>)
// CHECK: acc.exit_data wait_devnum([[I64VALUE]] : i64) wait([[I32VALUE]], [[IDXVALUE]] : i32, index) copyout([[ARGA]] : memref<10xf32>)
// -----


func.func @testenterdataop(%a: memref<10xf32>, %b: memref<10xf32>, %c: memref<10x10xf32>) -> () {
  %ifCond = arith.constant true
  %i64Value = arith.constant 1 : i64
  %i32Value = arith.constant 1 : i32
  %idxValue = arith.constant 1 : index

  acc.enter_data copyin(%a : memref<10xf32>)
  acc.enter_data create(%a : memref<10xf32>) create_zero(%b, %c : memref<10xf32>, memref<10x10xf32>)
  acc.enter_data attach(%a : memref<10xf32>)
  acc.enter_data copyin(%a : memref<10xf32>) attributes {async}
  acc.enter_data create(%a : memref<10xf32>) attributes {wait}
  acc.enter_data async(%i64Value : i64) copyin(%a : memref<10xf32>)
  acc.enter_data copyin(%a : memref<10xf32>) async(%i64Value : i64)
  acc.enter_data if(%ifCond) copyin(%a : memref<10xf32>)
  acc.enter_data wait_devnum(%i64Value: i64) wait(%i32Value, %idxValue : i32, index) copyin(%a : memref<10xf32>)

  return
}

// CHECK: func @testenterdataop([[ARGA:%.*]]: memref<10xf32>, [[ARGB:%.*]]: memref<10xf32>, [[ARGC:%.*]]: memref<10x10xf32>) {
// CHECK: [[IFCOND1:%.*]] = arith.constant true
// CHECK: [[I64VALUE:%.*]] = arith.constant 1 : i64
// CHECK: [[I32VALUE:%.*]] = arith.constant 1 : i32
// CHECK: [[IDXVALUE:%.*]] = arith.constant 1 : index
// CHECK: acc.enter_data copyin([[ARGA]] : memref<10xf32>)
// CHECK: acc.enter_data create([[ARGA]] : memref<10xf32>) create_zero([[ARGB]], [[ARGC]] : memref<10xf32>, memref<10x10xf32>)
// CHECK: acc.enter_data attach([[ARGA]] : memref<10xf32>)
// CHECK: acc.enter_data copyin([[ARGA]] : memref<10xf32>) attributes {async}
// CHECK: acc.enter_data create([[ARGA]] : memref<10xf32>) attributes {wait}
// CHECK: acc.enter_data async([[I64VALUE]] : i64) copyin([[ARGA]] : memref<10xf32>)
// CHECK: acc.enter_data async([[I64VALUE]] : i64) copyin([[ARGA]] : memref<10xf32>)
// CHECK: acc.enter_data if([[IFCOND]]) copyin([[ARGA]] : memref<10xf32>)
// CHECK: acc.enter_data wait_devnum([[I64VALUE]] : i64) wait([[I32VALUE]], [[IDXVALUE]] : i32, index) copyin([[ARGA]] : memref<10xf32>)
