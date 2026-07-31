// RUN: mlir-opt %s --normalize --mlir-use-nameloc-as-prefix 2>&1 | FileCheck %s

module {
  // CHECK-LABEL:   func.func @foo(
  // CHECK-SAME:                   %arg0: i32) -> i32 {
  // CHECK:           %vl13600$funcArg0-funcArg0$ = arith.addi %arg0, %arg0 : i32
  // CHECK:           return %vl13600$funcArg0-funcArg0$ : i32
  // CHECK:         }
  func.func @foo(%x: i32) -> i32 {
    %y = arith.addi %x, %x : i32
    return %y : i32
  }
  // CHECK-LABEL:   func.func @bar() -> i32 {
  // CHECK:           %vl10237$20b04$ = arith.constant 0 : i32
  // CHECK:           %vl15008$eafb0$ = arith.constant 4 : i32
  // CHECK:           %vl70789$vl10237-vl15008$ = arith.addi %vl10237$20b04$, %vl15008$eafb0$ : i32
  // CHECK:           %op27844$vl10237-vl70789$ = arith.addi %vl10237$20b04$, %vl70789$vl10237-vl15008$ : i32
  // CHECK:           %op27844$op27844-vl15008$ = arith.addi %op27844$vl10237-vl70789$, %vl15008$eafb0$ : i32
  // CHECK:           %vl16656$bf67a$ = memref.alloc() : memref<4xi32>
  // CHECK:           %vl77401$e527e$ = arith.constant 0 : index
  // CHECK:           memref.store %op27844$op27844-vl15008$, %vl16656$bf67a$[%vl77401$e527e$] : memref<4xi32>
  // CHECK:           %op15672$op27844-op27844$ = arith.addi %op27844$op27844-vl15008$, %op27844$vl10237-vl70789$ : i32
  // CHECK:           %vl70265$62631$ = arith.constant 1 : index
  // CHECK:           memref.store %op15672$op27844-op27844$, %vl16656$bf67a$[%vl70265$62631$] : memref<4xi32>
  // CHECK:           %op27844$vl10237-vl70789$_0 = arith.addi %vl10237$20b04$, %vl70789$vl10237-vl15008$ : i32
  // CHECK:           %op27844$op27844-vl15008$_1 = arith.addi %op27844$vl10237-vl70789$_0, %vl15008$eafb0$ : i32
  // CHECK:           %op15672$op27844-op27844$_2 = arith.addi %op27844$op27844-vl15008$_1, %op27844$vl10237-vl70789$_0 : i32
  // CHECK:           %op43149$vl16656-vl77401$ = memref.load %vl16656$bf67a$[%vl77401$e527e$] : memref<4xi32>
  // CHECK:           %op43149$vl16656-vl70265$ = memref.load %vl16656$bf67a$[%vl70265$62631$] : memref<4xi32>
  // CHECK:           %op12004$op43149-op43149$ = arith.addi %op43149$vl16656-vl70265$, %op43149$vl16656-vl77401$ : i32
  // CHECK:           %op52433foo$op15672$ = call @foo(%op15672$op27844-op27844$_2) : (i32) -> i32
  // CHECK:           %op91727$op12004-op52433$ = arith.addi %op12004$op43149-op43149$, %op52433foo$op15672$ : i32
  // CHECK:           return %op91727$op12004-op52433$ : i32
  // CHECK:         }

  func.func @bar() -> i32{
    %m0 = memref.alloc() : memref<4xi32>

    %c0 = arith.constant 4 : i32
    %zero = arith.constant 0 : i32
    
    %idx0 = arith.constant 0 : index
    %idx1 = arith.constant 1 : index

    %t  = arith.addi %zero, %c0 : i32

    %t2 = arith.addi %t, %zero : i32
    %t3 = arith.addi %t2, %c0 : i32
    %t4 = arith.addi %t3, %t2 : i32

    memref.store %t3, %m0[%idx0] : memref<4xi32>
    memref.store %t4, %m0[%idx1] : memref<4xi32>

    %a = memref.load %m0[%idx0] : memref<4xi32>
    %b = memref.load %m0[%idx1] : memref<4xi32>

    %t5 = arith.addi %t, %zero : i32
    %t6 = arith.addi %t5, %c0 : i32
    %t7 = arith.addi %t6, %t5 : i32
    %t8 = call @foo(%t7) : (i32) -> i32

    %t9 = arith.addi %a, %b : i32
    %t10 = arith.addi %t9, %t8 : i32
    return %t10 : i32
  }
}
