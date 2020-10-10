// RUN: mlir-opt -split-input-file %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: mlir-opt -split-input-file %s | mlir-opt -allow-unregistered-dialect  | FileCheck %s
// Verify the generic form can be parsed.
// RUN: mlir-opt -split-input-file -mlir-print-op-generic %s | mlir-opt -allow-unregistered-dialect | FileCheck %s

func @compute1(%A: memref<10x10xf32>, %B: memref<10x10xf32>, %C: memref<10x10xf32>) -> memref<10x10xf32> {
  %c0 = constant 0 : index
  %c10 = constant 10 : index
  %c1 = constant 1 : index
  %async = constant 1 : i64

  acc.parallel async(%async: i64) {
    acc.loop gang vector {
      scf.for %arg3 = %c0 to %c10 step %c1 {
        scf.for %arg4 = %c0 to %c10 step %c1 {
          scf.for %arg5 = %c0 to %c10 step %c1 {
            %a = load %A[%arg3, %arg5] : memref<10x10xf32>
            %b = load %B[%arg5, %arg4] : memref<10x10xf32>
            %cij = load %C[%arg3, %arg4] : memref<10x10xf32>
            %p = mulf %a, %b : f32
            %co = addf %cij, %p : f32
            store %co, %C[%arg3, %arg4] : memref<10x10xf32>
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
//  CHECK-NEXT:   %{{.*}} = constant 0 : index
//  CHECK-NEXT:   %{{.*}} = constant 10 : index
//  CHECK-NEXT:   %{{.*}} = constant 1 : index
//  CHECK-NEXT:   [[ASYNC:%.*]] = constant 1 : i64
//  CHECK-NEXT:   acc.parallel async([[ASYNC]]: i64) {
//  CHECK-NEXT:     acc.loop gang vector {
//  CHECK-NEXT:       scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//  CHECK-NEXT:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//  CHECK-NEXT:           scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//  CHECK-NEXT:             %{{.*}} = load %{{.*}}[%{{.*}}, %{{.*}}] : memref<10x10xf32>
//  CHECK-NEXT:             %{{.*}} = load %{{.*}}[%{{.*}}, %{{.*}}] : memref<10x10xf32>
//  CHECK-NEXT:             %{{.*}} = load %{{.*}}[%{{.*}}, %{{.*}}] : memref<10x10xf32>
//  CHECK-NEXT:             %{{.*}} = mulf %{{.*}}, %{{.*}} : f32
//  CHECK-NEXT:             %{{.*}} = addf %{{.*}}, %{{.*}} : f32
//  CHECK-NEXT:             store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<10x10xf32>
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

func @compute2(%A: memref<10x10xf32>, %B: memref<10x10xf32>, %C: memref<10x10xf32>) -> memref<10x10xf32> {
  %c0 = constant 0 : index
  %c10 = constant 10 : index
  %c1 = constant 1 : index

  acc.parallel {
    acc.loop {
      scf.for %arg3 = %c0 to %c10 step %c1 {
        scf.for %arg4 = %c0 to %c10 step %c1 {
          scf.for %arg5 = %c0 to %c10 step %c1 {
            %a = load %A[%arg3, %arg5] : memref<10x10xf32>
            %b = load %B[%arg5, %arg4] : memref<10x10xf32>
            %cij = load %C[%arg3, %arg4] : memref<10x10xf32>
            %p = mulf %a, %b : f32
            %co = addf %cij, %p : f32
            store %co, %C[%arg3, %arg4] : memref<10x10xf32>
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
//  CHECK-NEXT:   %{{.*}} = constant 0 : index
//  CHECK-NEXT:   %{{.*}} = constant 10 : index
//  CHECK-NEXT:   %{{.*}} = constant 1 : index
//  CHECK-NEXT:   acc.parallel {
//  CHECK-NEXT:     acc.loop {
//  CHECK-NEXT:       scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//  CHECK-NEXT:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//  CHECK-NEXT:           scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//  CHECK-NEXT:             %{{.*}} = load %{{.*}}[%{{.*}}, %{{.*}}] : memref<10x10xf32>
//  CHECK-NEXT:             %{{.*}} = load %{{.*}}[%{{.*}}, %{{.*}}] : memref<10x10xf32>
//  CHECK-NEXT:             %{{.*}} = load %{{.*}}[%{{.*}}, %{{.*}}] : memref<10x10xf32>
//  CHECK-NEXT:             %{{.*}} = mulf %{{.*}}, %{{.*}} : f32
//  CHECK-NEXT:             %{{.*}} = addf %{{.*}}, %{{.*}} : f32
//  CHECK-NEXT:             store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<10x10xf32>
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

func @compute3(%a: memref<10x10xf32>, %b: memref<10x10xf32>, %c: memref<10xf32>, %d: memref<10xf32>) -> memref<10xf32> {
  %lb = constant 0 : index
  %st = constant 1 : index
  %c10 = constant 10 : index
  %numGangs = constant 10 : i64
  %numWorkers = constant 10 : i64

  acc.data present(%a, %b, %c, %d: memref<10x10xf32>, memref<10x10xf32>, memref<10xf32>, memref<10xf32>) {
    acc.parallel num_gangs(%numGangs: i64) num_workers(%numWorkers: i64) private(%c : memref<10xf32>) {
      acc.loop gang {
        scf.for %x = %lb to %c10 step %st {
          acc.loop worker {
            scf.for %y = %lb to %c10 step %st {
              %axy = load %a[%x, %y] : memref<10x10xf32>
              %bxy = load %b[%x, %y] : memref<10x10xf32>
              %tmp = addf %axy, %bxy : f32
              store %tmp, %c[%y] : memref<10xf32>
            }
            acc.yield
          }

          acc.loop {
            // for i = 0 to 10 step 1
            //   d[x] += c[i]
            scf.for %i = %lb to %c10 step %st {
              %ci = load %c[%i] : memref<10xf32>
              %dx = load %d[%x] : memref<10xf32>
              %z = addf %ci, %dx : f32
              store %z, %d[%x] : memref<10xf32>
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
// CHECK-NEXT:   [[C0:%.*]] = constant 0 : index
// CHECK-NEXT:   [[C1:%.*]] = constant 1 : index
// CHECK-NEXT:   [[C10:%.*]] = constant 10 : index
// CHECK-NEXT:   [[NUMGANG:%.*]] = constant 10 : i64
// CHECK-NEXT:   [[NUMWORKERS:%.*]] = constant 10 : i64
// CHECK-NEXT:   acc.data present(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : memref<10x10xf32>, memref<10x10xf32>, memref<10xf32>, memref<10xf32>) {
// CHECK-NEXT:     acc.parallel num_gangs([[NUMGANG]]: i64) num_workers([[NUMWORKERS]]: i64) private([[ARG2]]: memref<10xf32>) {
// CHECK-NEXT:       acc.loop gang {
// CHECK-NEXT:         scf.for %{{.*}} = [[C0]] to [[C10]] step [[C1]] {
// CHECK-NEXT:           acc.loop worker {
// CHECK-NEXT:             scf.for %{{.*}} = [[C0]] to [[C10]] step [[C1]] {
// CHECK-NEXT:               %{{.*}} = load %{{.*}}[%{{.*}}, %{{.*}}] : memref<10x10xf32>
// CHECK-NEXT:               %{{.*}} = load %{{.*}}[%{{.*}}, %{{.*}}] : memref<10x10xf32>
// CHECK-NEXT:               %{{.*}} = addf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:               store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
// CHECK-NEXT:             }
// CHECK-NEXT:             acc.yield
// CHECK-NEXT:           }
// CHECK-NEXT:           acc.loop {
// CHECK-NEXT:             scf.for %{{.*}} = [[C0]] to [[C10]] step [[C1]] {
// CHECK-NEXT:               %{{.*}} = load %{{.*}}[%{{.*}}] : memref<10xf32>
// CHECK-NEXT:               %{{.*}} = load %{{.*}}[%{{.*}}] : memref<10xf32>
// CHECK-NEXT:               %{{.*}} = addf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:               store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
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

func @testloopop() -> () {
  %i64Value = constant 1 : i64
  %i32Value = constant 128 : i32
  %idxValue = constant 8 : index

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
  acc.loop tile(%i64Value: i64, %i64Value: i64) {
    "test.openacc_dummy_op"() : () -> ()
    acc.yield
  }
  acc.loop tile(%i32Value: i32, %i32Value: i32) {
    "test.openacc_dummy_op"() : () -> ()
    acc.yield
  }
  return
}

// CHECK:      [[I64VALUE:%.*]] = constant 1 : i64
// CHECK-NEXT: [[I32VALUE:%.*]] = constant 128 : i32
// CHECK-NEXT: [[IDXVALUE:%.*]] = constant 8 : index
// CHECK:      acc.loop gang worker vector {
// CHECK-NEXT:   "test.openacc_dummy_op"() : () -> ()
// CHECK-NEXT:   acc.yield
// CHECK-NEXT: }
// CHECK:      acc.loop gang(num=[[I64VALUE]]: i64) {
// CHECK-NEXT:   "test.openacc_dummy_op"() : () -> ()
// CHECK-NEXT:   acc.yield
// CHECK-NEXT: }
// CHECK:      acc.loop gang(static=[[I64VALUE]]: i64) {
// CHECK-NEXT:   "test.openacc_dummy_op"() : () -> ()
// CHECK-NEXT:   acc.yield
// CHECK-NEXT: }
// CHECK:      acc.loop worker([[I64VALUE]]: i64) {
// CHECK-NEXT:   "test.openacc_dummy_op"() : () -> ()
// CHECK-NEXT:   acc.yield
// CHECK-NEXT: }
// CHECK:      acc.loop worker([[I32VALUE]]: i32) {
// CHECK-NEXT:   "test.openacc_dummy_op"() : () -> ()
// CHECK-NEXT:   acc.yield
// CHECK-NEXT: }
// CHECK:      acc.loop worker([[IDXVALUE]]: index) {
// CHECK-NEXT:   "test.openacc_dummy_op"() : () -> ()
// CHECK-NEXT:   acc.yield
// CHECK-NEXT: }
// CHECK:      acc.loop vector([[I64VALUE]]: i64) {
// CHECK-NEXT:   "test.openacc_dummy_op"() : () -> ()
// CHECK-NEXT:   acc.yield
// CHECK-NEXT: }
// CHECK:      acc.loop vector([[I32VALUE]]: i32) {
// CHECK-NEXT:   "test.openacc_dummy_op"() : () -> ()
// CHECK-NEXT:   acc.yield
// CHECK-NEXT: }
// CHECK:      acc.loop vector([[IDXVALUE]]: index) {
// CHECK-NEXT:   "test.openacc_dummy_op"() : () -> ()
// CHECK-NEXT:   acc.yield
// CHECK-NEXT: }
// CHECK:      acc.loop gang(num=[[I64VALUE]]: i64) worker vector {
// CHECK-NEXT:   "test.openacc_dummy_op"() : () -> ()
// CHECK-NEXT:   acc.yield
// CHECK-NEXT: }
// CHECK:      acc.loop gang(num=[[I64VALUE]]: i64, static=[[I64VALUE]]: i64) worker([[I64VALUE]]: i64) vector([[I64VALUE]]: i64) {
// CHECK-NEXT:   "test.openacc_dummy_op"() : () -> ()
// CHECK-NEXT:   acc.yield
// CHECK-NEXT: }
// CHECK:      acc.loop gang(num=[[I32VALUE]]: i32, static=[[IDXVALUE]]: index) {
// CHECK-NEXT:   "test.openacc_dummy_op"() : () -> ()
// CHECK-NEXT:   acc.yield
// CHECK-NEXT: }
// CHECK:      acc.loop tile([[I64VALUE]]: i64, [[I64VALUE]]: i64) {
// CHECK-NEXT:   "test.openacc_dummy_op"() : () -> ()
// CHECK-NEXT:   acc.yield
// CHECK-NEXT: }
// CHECK:      acc.loop tile([[I32VALUE]]: i32, [[I32VALUE]]: i32) {
// CHECK-NEXT:   "test.openacc_dummy_op"() : () -> ()
// CHECK-NEXT:   acc.yield
// CHECK-NEXT: }

// -----

func @testparallelop(%a: memref<10xf32>, %b: memref<10xf32>, %c: memref<10x10xf32>) -> () {
  %i64value = constant 1 : i64
  %i32value = constant 1 : i32
  %idxValue = constant 1 : index
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
  acc.parallel wait(%i64value: i64, %i32value: i32, %idxValue: index) {
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
  acc.parallel copyin(%a: memref<10xf32>, %b: memref<10xf32>) {
  }
  acc.parallel copyin_readonly(%a: memref<10xf32>, %b: memref<10xf32>) {
  }
  acc.parallel copyin(%a: memref<10xf32>) copyout_zero(%b: memref<10xf32>, %c: memref<10x10xf32>) {
  }
  acc.parallel copyout(%b: memref<10xf32>, %c: memref<10x10xf32>) create(%a: memref<10xf32>) {
  }
  acc.parallel copyout_zero(%b: memref<10xf32>, %c: memref<10x10xf32>) create_zero(%a: memref<10xf32>) {
  }
  acc.parallel no_create(%a: memref<10xf32>) present(%b: memref<10xf32>, %c: memref<10x10xf32>) {
  }
  acc.parallel deviceptr(%a: memref<10xf32>) attach(%b: memref<10xf32>, %c: memref<10x10xf32>) {
  }
  acc.parallel private(%a: memref<10xf32>, %c: memref<10x10xf32>) firstprivate(%b: memref<10xf32>) {
  }
  acc.parallel {
  } attributes {defaultAttr = "none"}
  acc.parallel {
  } attributes {defaultAttr = "present"}
  acc.parallel {
  } attributes {asyncAttr}
  acc.parallel {
  } attributes {waitAttr}
  acc.parallel {
  } attributes {selfAttr}
  return
}

// CHECK:      func @testparallelop([[ARGA:%.*]]: memref<10xf32>, [[ARGB:%.*]]: memref<10xf32>, [[ARGC:%.*]]: memref<10x10xf32>) {
// CHECK:      [[I64VALUE:%.*]] = constant 1 : i64
// CHECK:      [[I32VALUE:%.*]] = constant 1 : i32
// CHECK:      [[IDXVALUE:%.*]] = constant 1 : index
// CHECK:      acc.parallel async([[I64VALUE]]: i64) {
// CHECK-NEXT: }
// CHECK:      acc.parallel async([[I32VALUE]]: i32) {
// CHECK-NEXT: }
// CHECK:      acc.parallel async([[IDXVALUE]]: index) {
// CHECK-NEXT: }
// CHECK:      acc.parallel wait([[I64VALUE]]: i64) {
// CHECK-NEXT: }
// CHECK:      acc.parallel wait([[I32VALUE]]: i32) {
// CHECK-NEXT: }
// CHECK:      acc.parallel wait([[IDXVALUE]]: index) {
// CHECK-NEXT: }
// CHECK:      acc.parallel wait([[I64VALUE]]: i64, [[I32VALUE]]: i32, [[IDXVALUE]]: index) {
// CHECK-NEXT: }
// CHECK:      acc.parallel num_gangs([[I64VALUE]]: i64) {
// CHECK-NEXT: }
// CHECK:      acc.parallel num_gangs([[I32VALUE]]: i32) {
// CHECK-NEXT: }
// CHECK:      acc.parallel num_gangs([[IDXVALUE]]: index) {
// CHECK-NEXT: }
// CHECK:      acc.parallel num_workers([[I64VALUE]]: i64) {
// CHECK-NEXT: }
// CHECK:      acc.parallel num_workers([[I32VALUE]]: i32) {
// CHECK-NEXT: }
// CHECK:      acc.parallel num_workers([[IDXVALUE]]: index) {
// CHECK-NEXT: }
// CHECK:      acc.parallel vector_length([[I64VALUE]]: i64) {
// CHECK-NEXT: }
// CHECK:      acc.parallel vector_length([[I32VALUE]]: i32) {
// CHECK-NEXT: }
// CHECK:      acc.parallel vector_length([[IDXVALUE]]: index) {
// CHECK-NEXT: }
// CHECK:      acc.parallel copyin([[ARGA]]: memref<10xf32>, [[ARGB]]: memref<10xf32>) {
// CHECK-NEXT: }
// CHECK:      acc.parallel copyin_readonly([[ARGA]]: memref<10xf32>, [[ARGB]]: memref<10xf32>) {
// CHECK-NEXT: }
// CHECK:      acc.parallel copyin([[ARGA]]: memref<10xf32>) copyout_zero([[ARGB]]: memref<10xf32>, [[ARGC]]: memref<10x10xf32>) {
// CHECK-NEXT: }
// CHECK:      acc.parallel copyout([[ARGB]]: memref<10xf32>, [[ARGC]]: memref<10x10xf32>) create([[ARGA]]: memref<10xf32>) {
// CHECK-NEXT: }
// CHECK:      acc.parallel copyout_zero([[ARGB]]: memref<10xf32>, [[ARGC]]: memref<10x10xf32>) create_zero([[ARGA]]: memref<10xf32>) {
// CHECK-NEXT: }
// CHECK:      acc.parallel no_create([[ARGA]]: memref<10xf32>) present([[ARGB]]: memref<10xf32>, [[ARGC]]: memref<10x10xf32>) {
// CHECK-NEXT: }
// CHECK:      acc.parallel deviceptr([[ARGA]]: memref<10xf32>) attach([[ARGB]]: memref<10xf32>, [[ARGC]]: memref<10x10xf32>) {
// CHECK-NEXT: }
// CHECK:      acc.parallel private([[ARGA]]: memref<10xf32>, [[ARGC]]: memref<10x10xf32>) firstprivate([[ARGB]]: memref<10xf32>) {
// CHECK-NEXT: }
// CHECK:      acc.parallel {
// CHECK-NEXT: } attributes {defaultAttr = "none"}
// CHECK:      acc.parallel {
// CHECK-NEXT: } attributes {defaultAttr = "present"}
// CHECK:      acc.parallel {
// CHECK-NEXT: } attributes {asyncAttr}
// CHECK:      acc.parallel {
// CHECK-NEXT: } attributes {waitAttr}
// CHECK:      acc.parallel {
// CHECK-NEXT: } attributes {selfAttr}

// -----

func @testdataop(%a: memref<10xf32>, %b: memref<10xf32>, %c: memref<10x10xf32>) -> () {
  %ifCond = constant true
  acc.data if(%ifCond) present(%a : memref<10xf32>) {
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
  } attributes { defaultAttr = "none" }
  acc.data present(%a : memref<10xf32>) {
  } attributes { defaultAttr = "present" }
  acc.data {
  } attributes { defaultAttr = "none" }
  return
}

// CHECK:      func @testdataop([[ARGA:%.*]]: memref<10xf32>, [[ARGB:%.*]]: memref<10xf32>, [[ARGC:%.*]]: memref<10x10xf32>) {
// CHECK:      [[IFCOND1:%.*]] = constant true
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
// CHECK-NEXT: } attributes {defaultAttr = "none"}
// CHECK:      acc.data present([[ARGA]] : memref<10xf32>) {
// CHECK-NEXT: } attributes {defaultAttr = "present"}
// CHECK:      acc.data {
// CHECK-NEXT: } attributes {defaultAttr = "none"}

// -----

func @testupdateop(%a: memref<10xf32>, %b: memref<10xf32>, %c: memref<10x10xf32>) -> () {
  %i64Value = constant 1 : i64
  %i32Value = constant 1 : i32
  %idxValue = constant 1 : index
  acc.update async(%i64Value: i64) host(%a: memref<10xf32>)
  acc.update async(%i32Value: i32) host(%a: memref<10xf32>)
  acc.update async(%idxValue: index) host(%a: memref<10xf32>)
  acc.update wait_devnum(%i64Value: i64) wait(%i32Value, %idxValue : i32, index) host(%a: memref<10xf32>)
  acc.update host(%a: memref<10xf32>) device(%b, %c : memref<10xf32>, memref<10x10xf32>)
  acc.update host(%a: memref<10xf32>) device(%b, %c : memref<10xf32>, memref<10x10xf32>) attributes {async}
  acc.update host(%a: memref<10xf32>) device(%b, %c : memref<10xf32>, memref<10x10xf32>) attributes {wait}
  acc.update host(%a: memref<10xf32>) device(%b, %c : memref<10xf32>, memref<10x10xf32>) attributes {ifPresent}
  return
}

// CHECK: func @testupdateop([[ARGA:%.*]]: memref<10xf32>, [[ARGB:%.*]]: memref<10xf32>, [[ARGC:%.*]]: memref<10x10xf32>) {
// CHECK:   [[I64VALUE:%.*]] = constant 1 : i64
// CHECK:   [[I32VALUE:%.*]] = constant 1 : i32
// CHECK:   [[IDXVALUE:%.*]] = constant 1 : index
// CHECK:   acc.update async([[I64VALUE]] : i64) host([[ARGA]] : memref<10xf32>)
// CHECK:   acc.update async([[I32VALUE]] : i32) host([[ARGA]] : memref<10xf32>)
// CHECK:   acc.update async([[IDXVALUE]] : index) host([[ARGA]] : memref<10xf32>)
// CHECK:   acc.update wait_devnum([[I64VALUE]] : i64) wait([[I32VALUE]], [[IDXVALUE]] : i32, index) host([[ARGA]] : memref<10xf32>)
// CHECK:   acc.update host([[ARGA]] : memref<10xf32>) device([[ARGB]], [[ARGC]] : memref<10xf32>, memref<10x10xf32>)
// CHECK:   acc.update host([[ARGA]] : memref<10xf32>) device([[ARGB]], [[ARGC]] : memref<10xf32>, memref<10x10xf32>) attributes {async}
// CHECK:   acc.update host([[ARGA]] : memref<10xf32>) device([[ARGB]], [[ARGC]] : memref<10xf32>, memref<10x10xf32>) attributes {wait}
// CHECK:   acc.update host([[ARGA]] : memref<10xf32>) device([[ARGB]], [[ARGC]] : memref<10xf32>, memref<10x10xf32>) attributes {ifPresent}

// -----

%i64Value = constant 1 : i64
%i32Value = constant 1 : i32
%idxValue = constant 1 : index
%ifCond = constant true
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
acc.wait if(%ifCond)

// CHECK: [[I64VALUE:%.*]] = constant 1 : i64
// CHECK: [[I32VALUE:%.*]] = constant 1 : i32
// CHECK: [[IDXVALUE:%.*]] = constant 1 : index
// CHECK: [[IFCOND:%.*]] = constant true
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
// CHECK: acc.wait if([[IFCOND]])

// -----

%i64Value = constant 1 : i64
%i32Value = constant 1 : i32
%i32Value2 = constant 2 : i32
%idxValue = constant 1 : index
%ifCond = constant true
acc.init
acc.init device_type(%i32Value : i32)
acc.init device_type(%i32Value, %i32Value2 : i32, i32)
acc.init device_num(%i64Value : i64)
acc.init device_num(%i32Value : i32)
acc.init device_num(%idxValue : index)
acc.init if(%ifCond)

// CHECK: [[I64VALUE:%.*]] = constant 1 : i64
// CHECK: [[I32VALUE:%.*]] = constant 1 : i32
// CHECK: [[I32VALUE2:%.*]] = constant 2 : i32
// CHECK: [[IDXVALUE:%.*]] = constant 1 : index
// CHECK: [[IFCOND:%.*]] = constant true
// CHECK: acc.init
// CHECK: acc.init device_type([[I32VALUE]] : i32)
// CHECK: acc.init device_type([[I32VALUE]], [[I32VALUE2]] : i32, i32)
// CHECK: acc.init device_num([[I64VALUE]] : i64)
// CHECK: acc.init device_num([[I32VALUE]] : i32)
// CHECK: acc.init device_num([[IDXVALUE]] : index)
// CHECK: acc.init if([[IFCOND]])

// -----

%i64Value = constant 1 : i64
%i32Value = constant 1 : i32
%i32Value2 = constant 2 : i32
%idxValue = constant 1 : index
%ifCond = constant true
acc.shutdown
acc.shutdown device_type(%i32Value : i32)
acc.shutdown device_type(%i32Value, %i32Value2 : i32, i32)
acc.shutdown device_num(%i64Value : i64)
acc.shutdown device_num(%i32Value : i32)
acc.shutdown device_num(%idxValue : index)
acc.shutdown if(%ifCond)

// CHECK: [[I64VALUE:%.*]] = constant 1 : i64
// CHECK: [[I32VALUE:%.*]] = constant 1 : i32
// CHECK: [[I32VALUE2:%.*]] = constant 2 : i32
// CHECK: [[IDXVALUE:%.*]] = constant 1 : index
// CHECK: [[IFCOND:%.*]] = constant true
// CHECK: acc.shutdown
// CHECK: acc.shutdown device_type([[I32VALUE]] : i32)
// CHECK: acc.shutdown device_type([[I32VALUE]], [[I32VALUE2]] : i32, i32)
// CHECK: acc.shutdown device_num([[I64VALUE]] : i64)
// CHECK: acc.shutdown device_num([[I32VALUE]] : i32)
// CHECK: acc.shutdown device_num([[IDXVALUE]] : index)
// CHECK: acc.shutdown if([[IFCOND]])

// -----

func @testexitdataop(%a: memref<10xf32>, %b: memref<10xf32>, %c: memref<10x10xf32>) -> () {
  %ifCond = constant true
  %i64Value = constant 1 : i64
  %i32Value = constant 1 : i32
  %idxValue = constant 1 : index

  acc.exit_data copyout(%a : memref<10xf32>)
  acc.exit_data delete(%a : memref<10xf32>)
  acc.exit_data delete(%a : memref<10xf32>) attributes {async,finalize}
  acc.exit_data detach(%a : memref<10xf32>)
  acc.exit_data copyout(%a : memref<10xf32>) attributes {async}
  acc.exit_data delete(%a : memref<10xf32>) attributes {wait}
  acc.exit_data async(%i64Value : i64) copyout(%a : memref<10xf32>)
  acc.exit_data if(%ifCond) copyout(%a : memref<10xf32>)
  acc.exit_data wait_devnum(%i64Value: i64) wait(%i32Value, %idxValue : i32, index) copyout(%a : memref<10xf32>)

  return
}

// CHECK: func @testexitdataop([[ARGA:%.*]]: memref<10xf32>, [[ARGB:%.*]]: memref<10xf32>, [[ARGC:%.*]]: memref<10x10xf32>) {
// CHECK: [[IFCOND1:%.*]] = constant true
// CHECK: [[I64VALUE:%.*]] = constant 1 : i64
// CHECK: [[I32VALUE:%.*]] = constant 1 : i32
// CHECK: [[IDXVALUE:%.*]] = constant 1 : index
// CHECK: acc.exit_data copyout([[ARGA]] : memref<10xf32>)
// CHECK: acc.exit_data delete([[ARGA]] : memref<10xf32>)
// CHECK: acc.exit_data delete([[ARGA]] : memref<10xf32>) attributes {async, finalize}
// CHECK: acc.exit_data detach([[ARGA]] : memref<10xf32>)
// CHECK: acc.exit_data copyout([[ARGA]] : memref<10xf32>) attributes {async}
// CHECK: acc.exit_data delete([[ARGA]] : memref<10xf32>) attributes {wait}
// CHECK: acc.exit_data async([[I64VALUE]] : i64) copyout([[ARGA]] : memref<10xf32>)
// CHECK: acc.exit_data if([[IFCOND]]) copyout([[ARGA]] : memref<10xf32>)
// CHECK: acc.exit_data wait_devnum([[I64VALUE]] : i64) wait([[I32VALUE]], [[IDXVALUE]] : i32, index) copyout([[ARGA]] : memref<10xf32>)
