// RUN: mlir-opt %s --convert-arith-to-smt | FileCheck %s

// CHECK-LABEL: func @test
// CHECK-SAME: ([[A0:%.+]]: !smt.bv<32>, [[A1:%.+]]: !smt.bv<32>, [[A2:%.+]]: !smt.bv<32>, [[A3:%.+]]: !smt.bv<32>, [[A4:%.+]]: !smt.bv<1>, [[ARG5:%.+]]: !smt.bv<4>)
func.func @test(%a0: !smt.bv<32>, %a1: !smt.bv<32>, %a2: !smt.bv<32>, %a3: !smt.bv<32>, %a4: !smt.bv<1>, %a5: !smt.bv<4>) {
  %arg0 = builtin.unrealized_conversion_cast %a0 : !smt.bv<32> to i32
  %arg1 = builtin.unrealized_conversion_cast %a1 : !smt.bv<32> to i32
  %arg2 = builtin.unrealized_conversion_cast %a2 : !smt.bv<32> to i32
  %arg3 = builtin.unrealized_conversion_cast %a3 : !smt.bv<32> to i32
  %arg4 = builtin.unrealized_conversion_cast %a4 : !smt.bv<1> to i1
  %arg5 = builtin.unrealized_conversion_cast %a5 : !smt.bv<4> to i4

  // CHECK:      [[ZERO:%.+]] = smt.bv.constant #smt.bv<0> : !smt.bv<32>
  // CHECK-NEXT: [[IS_ZERO:%.+]] = smt.eq [[A1]], [[ZERO]] : !smt.bv<32>
  // CHECK-NEXT: [[UNDEF:%.+]] = smt.declare_fun : !smt.bv<32>
  // CHECK-NEXT: [[DIV:%.+]] = smt.bv.sdiv [[A0]], [[A1]] : !smt.bv<32>
  // CHECK-NEXT: smt.ite [[IS_ZERO]], [[UNDEF]], [[DIV]] : !smt.bv<32>
  %0 = arith.divsi %arg0, %arg1 : i32
  // CHECK-NEXT: [[ZERO:%.+]] = smt.bv.constant #smt.bv<0> : !smt.bv<32>
  // CHECK-NEXT: [[IS_ZERO:%.+]] = smt.eq [[A1]], [[ZERO]] : !smt.bv<32>
  // CHECK-NEXT: [[UNDEF:%.+]] = smt.declare_fun : !smt.bv<32>
  // CHECK-NEXT: [[DIV:%.+]] = smt.bv.udiv [[A0]], [[A1]] : !smt.bv<32>
  // CHECK-NEXT: smt.ite [[IS_ZERO]], [[UNDEF]], [[DIV]] : !smt.bv<32>
  %1 = arith.divui %arg0, %arg1 : i32
  // CHECK-NEXT: [[ZERO:%.+]] = smt.bv.constant #smt.bv<0> : !smt.bv<32>
  // CHECK-NEXT: [[IS_ZERO:%.+]] = smt.eq [[A1]], [[ZERO]] : !smt.bv<32>
  // CHECK-NEXT: [[UNDEF:%.+]] = smt.declare_fun : !smt.bv<32>
  // CHECK-NEXT: [[DIV:%.+]] = smt.bv.srem [[A0]], [[A1]] : !smt.bv<32>
  // CHECK-NEXT: smt.ite [[IS_ZERO]], [[UNDEF]], [[DIV]] : !smt.bv<32>
  %2 = arith.remsi %arg0, %arg1 : i32
  // CHECK-NEXT: [[ZERO:%.+]] = smt.bv.constant #smt.bv<0> : !smt.bv<32>
  // CHECK-NEXT: [[IS_ZERO:%.+]] = smt.eq [[A1]], [[ZERO]] : !smt.bv<32>
  // CHECK-NEXT: [[UNDEF:%.+]] = smt.declare_fun : !smt.bv<32>
  // CHECK-NEXT: [[DIV:%.+]] = smt.bv.urem [[A0]], [[A1]] : !smt.bv<32>
  // CHECK-NEXT: smt.ite [[IS_ZERO]], [[UNDEF]], [[DIV]] : !smt.bv<32>
  %3 = arith.remui %arg0, %arg1 : i32

  // CHECK-NEXT: [[NEG:%.+]] = smt.bv.neg [[A1]] : !smt.bv<32>
  // CHECK-NEXT: smt.bv.add [[A0]], [[NEG]] : !smt.bv<32>
  %7 = arith.subi %arg0, %arg1 : i32

  // CHECK-NEXT: [[A5:%.+]] = smt.bv.add [[A0]], [[A1]] : !smt.bv<32>
  %8 = arith.addi %arg0, %arg1 : i32
  // CHECK-NEXT: [[B1:%.+]] = smt.bv.mul [[A0]], [[A1]] : !smt.bv<32>
  %9 = arith.muli %arg0, %arg1 : i32
  // CHECK-NEXT: [[C1:%.+]] = smt.bv.and [[A0]], [[A1]] : !smt.bv<32>
  %10 = arith.andi %arg0, %arg1 : i32
  // CHECK-NEXT: [[D1:%.+]] = smt.bv.or [[A0]], [[A1]] : !smt.bv<32>
  %11 = arith.ori %arg0, %arg1 : i32
  // CHECK-NEXT: [[E1:%.+]] = smt.bv.xor [[A0]], [[A1]] : !smt.bv<32>
  %12 = arith.xori %arg0, %arg1 : i32

  // CHECK-NEXT: smt.eq [[A0]], [[A1]] : !smt.bv<32>
  %14 = arith.cmpi eq, %arg0, %arg1 : i32
  // CHECK-NEXT: smt.distinct [[A0]], [[A1]] : !smt.bv<32>
  %15 = arith.cmpi ne, %arg0, %arg1 : i32
  // CHECK-NEXT: smt.bv.cmp sle [[A0]], [[A1]] : !smt.bv<32>
  %20 = arith.cmpi sle, %arg0, %arg1 : i32
  // CHECK-NEXT: smt.bv.cmp slt [[A0]], [[A1]] : !smt.bv<32>
  %21 = arith.cmpi slt, %arg0, %arg1 : i32
  // CHECK-NEXT: smt.bv.cmp ule [[A0]], [[A1]] : !smt.bv<32>
  %22 = arith.cmpi ule, %arg0, %arg1 : i32
  // CHECK-NEXT: smt.bv.cmp ult [[A0]], [[A1]] : !smt.bv<32>
  %23 = arith.cmpi ult, %arg0, %arg1 : i32
  // CHECK-NEXT: smt.bv.cmp sge [[A0]], [[A1]] : !smt.bv<32>
  %24 = arith.cmpi sge, %arg0, %arg1 : i32
  // CHECK-NEXT: smt.bv.cmp sgt [[A0]], [[A1]] : !smt.bv<32>
  %25 = arith.cmpi sgt, %arg0, %arg1 : i32
  // CHECK-NEXT: smt.bv.cmp uge [[A0]], [[A1]] : !smt.bv<32>
  %26 = arith.cmpi uge, %arg0, %arg1 : i32
  // CHECK-NEXT: smt.bv.cmp ugt [[A0]], [[A1]] : !smt.bv<32>
  %27 = arith.cmpi ugt, %arg0, %arg1 : i32

  // CHECK-NEXT: %{{.*}} = smt.bv.shl [[A0]], [[A1]] : !smt.bv<32>
  %32 = arith.shli %arg0, %arg1 : i32
  // CHECK-NEXT: %{{.*}} = smt.bv.ashr [[A0]], [[A1]] : !smt.bv<32>
  %33 = arith.shrsi %arg0, %arg1 : i32
  // CHECK-NEXT: %{{.*}} = smt.bv.lshr [[A0]], [[A1]] : !smt.bv<32>
  %34 = arith.shrui %arg0, %arg1 : i32

  // The arith.cmpi folder is called before the conversion patterns and produces
  // a `arith.constant` operation.
  // CHECK-NEXT: smt.bv.constant #smt.bv<-1> : !smt.bv<1>
  %35 = arith.cmpi eq, %arg0, %arg0 : i32

  return
}
