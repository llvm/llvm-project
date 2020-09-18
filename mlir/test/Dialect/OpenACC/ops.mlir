// RUN: mlir-opt -allow-unregistered-dialect %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: mlir-opt -allow-unregistered-dialect %s | mlir-opt -allow-unregistered-dialect  | FileCheck %s
// Verify the generic form can be parsed.
// RUN: mlir-opt -allow-unregistered-dialect -mlir-print-op-generic %s | mlir-opt -allow-unregistered-dialect | FileCheck %s

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


func @compute3(%a: memref<10x10xf32>, %b: memref<10x10xf32>, %c: memref<10xf32>, %d: memref<10xf32>) -> memref<10xf32> {
  %lb = constant 0 : index
  %st = constant 1 : index
  %c10 = constant 10 : index
  %numGangs = constant 10 : i64
  %numWorkers = constant 10 : i64

  acc.data present(%a: memref<10x10xf32>, %b: memref<10x10xf32>, %c: memref<10xf32>, %d: memref<10xf32>) {
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
// CHECK-NEXT:   acc.data present(%{{.*}}: memref<10x10xf32>, %{{.*}}: memref<10x10xf32>, %{{.*}}: memref<10xf32>, %{{.*}}: memref<10xf32>) {
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

func @testop(%a: memref<10xf32>) -> () {
  %workerNum = constant 1 : i64
  %vectorLength = constant 128 : i64
  %gangNum = constant 8 : i64
  %gangStatic = constant 2 : i64
  %tileSize = constant 2 : i64
  acc.loop gang worker vector {
    "some.op"() : () -> ()
    acc.yield
  }
  acc.loop gang(num: %gangNum) {
    "some.op"() : () -> ()
    acc.yield
  }
  acc.loop gang(static: %gangStatic) {
    "some.op"() : () -> ()
    acc.yield
  }
  acc.loop worker(%workerNum) {
    "some.op"() : () -> ()
    acc.yield
  }
  acc.loop vector(%vectorLength) {
    "some.op"() : () -> ()
    acc.yield
  }
  acc.loop gang(num: %gangNum) worker vector {
    "some.op"() : () -> ()
    acc.yield
  }
  acc.loop gang(num: %gangNum, static: %gangStatic) worker(%workerNum) vector(%vectorLength) {
    "some.op"() : () -> ()
    acc.yield
  }
  acc.loop tile(%tileSize : i64, %tileSize : i64) {
    "some.op"() : () -> ()
    acc.yield
  }
  return
}

// CHECK:      [[WORKERNUM:%.*]] = constant 1 : i64
// CHECK-NEXT: [[VECTORLENGTH:%.*]] = constant 128 : i64
// CHECK-NEXT: [[GANGNUM:%.*]] = constant 8 : i64
// CHECK-NEXT: [[GANGSTATIC:%.*]] = constant 2 : i64
// CHECK-NEXT: [[TILESIZE:%.*]] = constant 2 : i64
// CHECK-NEXT: acc.loop gang worker vector {
// CHECK-NEXT:   "some.op"() : () -> ()
// CHECK-NEXT:   acc.yield
// CHECK-NEXT: }
// CHECK-NEXT: acc.loop gang(num: [[GANGNUM]]) {
// CHECK-NEXT:   "some.op"() : () -> ()
// CHECK-NEXT:   acc.yield
// CHECK-NEXT: }
// CHECK-NEXT: acc.loop gang(static: [[GANGSTATIC]]) {
// CHECK-NEXT:   "some.op"() : () -> ()
// CHECK-NEXT:   acc.yield
// CHECK-NEXT: }
// CHECK-NEXT: acc.loop worker([[WORKERNUM]]) {
// CHECK-NEXT:   "some.op"() : () -> ()
// CHECK-NEXT:   acc.yield
// CHECK-NEXT: }
// CHECK-NEXT: acc.loop vector([[VECTORLENGTH]]) {
// CHECK-NEXT:   "some.op"() : () -> ()
// CHECK-NEXT:   acc.yield
// CHECK-NEXT: }
// CHECK-NEXT: acc.loop gang(num: [[GANGNUM]]) worker vector {
// CHECK-NEXT:   "some.op"() : () -> ()
// CHECK-NEXT:   acc.yield
// CHECK-NEXT: }
// CHECK-NEXT: acc.loop gang(num: [[GANGNUM]], static: [[GANGSTATIC]]) worker([[WORKERNUM]]) vector([[VECTORLENGTH]]) {
// CHECK-NEXT:   "some.op"() : () -> ()
// CHECK-NEXT:   acc.yield
// CHECK-NEXT: }
// CHECK-NEXT: acc.loop tile([[TILESIZE]]: i64, [[TILESIZE]]: i64) {
// CHECK-NEXT:   "some.op"() : () -> ()
// CHECK-NEXT:   acc.yield
// CHECK-NEXT: }


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
