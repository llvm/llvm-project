// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// CHECK: irdl.dialect @testvar {
irdl.dialect @testvar {

  // CHECK-LABEL: irdl.operation @single_operand {
  // CHECK-NEXT:    %[[v0:[^ ]*]] = irdl.is i32
  // CHECK-NEXT:    irdl.operands(foo: %[[v0]])
  // CHECK-NEXT:  }
  irdl.operation @single_operand {
    %0 = irdl.is i32
    irdl.operands(foo: single %0)
  }

  // CHECK-LABEL: irdl.operation @var_operand {
  // CHECK-NEXT:    %[[v0:[^ ]*]] = irdl.is i16 
  // CHECK-NEXT:    %[[v1:[^ ]*]] = irdl.is i32 
  // CHECK-NEXT:    %[[v2:[^ ]*]] = irdl.is i64 
  // CHECK-NEXT:    irdl.operands(foo: %[[v0]], bar: variadic %[[v1]], baz: %[[v2]])
  // CHECK-NEXT:  }
  irdl.operation @var_operand {
    %0 = irdl.is i16
    %1 = irdl.is i32
    %2 = irdl.is i64
    irdl.operands(foo: %0, bar: variadic %1, baz: %2)
  }

  // CHECK-LABEL: irdl.operation @opt_operand {
  // CHECK-NEXT:    %[[v0:[^ ]*]] = irdl.is i16 
  // CHECK-NEXT:    %[[v1:[^ ]*]] = irdl.is i32 
  // CHECK-NEXT:    %[[v2:[^ ]*]] = irdl.is i64 
  // CHECK-NEXT:    irdl.operands(foo: %[[v0]], bar: optional %[[v1]], baz: %[[v2]])
  // CHECK-NEXT:  }
  irdl.operation @opt_operand {
    %0 = irdl.is i16
    %1 = irdl.is i32
    %2 = irdl.is i64
    irdl.operands(foo: %0, bar: optional %1, baz: %2)
  }

  // CHECK-LABEL: irdl.operation @var_and_opt_operand {
  // CHECK-NEXT:    %[[v0:[^ ]*]] = irdl.is i16 
  // CHECK-NEXT:    %[[v1:[^ ]*]] = irdl.is i32 
  // CHECK-NEXT:    %[[v2:[^ ]*]] = irdl.is i64 
  // CHECK-NEXT:    irdl.operands(foo: variadic %[[v0]], bar: optional %[[v1]], baz: %[[v2]])
  // CHECK-NEXT:  }
  irdl.operation @var_and_opt_operand {
    %0 = irdl.is i16
    %1 = irdl.is i32
    %2 = irdl.is i64
    irdl.operands(foo: variadic %0, bar: optional %1, baz: %2)
  }


  // CHECK-LABEL: irdl.operation @single_result {
  // CHECK-NEXT:    %[[v0:[^ ]*]] = irdl.is i32
  // CHECK-NEXT:    irdl.results(foo: %[[v0]])
  // CHECK-NEXT:  }
  irdl.operation @single_result {
    %0 = irdl.is i32
    irdl.results(foo: single %0)
  }

  // CHECK-LABEL: irdl.operation @var_result {
  // CHECK-NEXT:    %[[v0:[^ ]*]] = irdl.is i16 
  // CHECK-NEXT:    %[[v1:[^ ]*]] = irdl.is i32 
  // CHECK-NEXT:    %[[v2:[^ ]*]] = irdl.is i64 
  // CHECK-NEXT:    irdl.results(foo: %[[v0]], bar: variadic %[[v1]], baz: %[[v2]])
  // CHECK-NEXT:  }
  irdl.operation @var_result {
    %0 = irdl.is i16
    %1 = irdl.is i32
    %2 = irdl.is i64
    irdl.results(foo: %0, bar: variadic %1, baz: %2)
  }

  // CHECK-LABEL: irdl.operation @opt_result {
  // CHECK-NEXT:    %[[v0:[^ ]*]] = irdl.is i16 
  // CHECK-NEXT:    %[[v1:[^ ]*]] = irdl.is i32 
  // CHECK-NEXT:    %[[v2:[^ ]*]] = irdl.is i64 
  // CHECK-NEXT:    irdl.results(foo: %[[v0]], bar: optional %[[v1]], baz: %[[v2]])
  // CHECK-NEXT:  }
  irdl.operation @opt_result {
    %0 = irdl.is i16
    %1 = irdl.is i32
    %2 = irdl.is i64
    irdl.results(foo: %0, bar: optional %1, baz: %2)
  }

  // CHECK-LABEL: irdl.operation @var_and_opt_result {
  // CHECK-NEXT:    %[[v0:[^ ]*]] = irdl.is i16 
  // CHECK-NEXT:    %[[v1:[^ ]*]] = irdl.is i32 
  // CHECK-NEXT:    %[[v2:[^ ]*]] = irdl.is i64 
  // CHECK-NEXT:    irdl.results(foo: variadic %[[v0]], bar: optional %[[v1]], baz: %[[v2]])
  // CHECK-NEXT:  }
  irdl.operation @var_and_opt_result {
    %0 = irdl.is i16
    %1 = irdl.is i32
    %2 = irdl.is i64
    irdl.results(foo: variadic %0, bar: optional %1, baz: %2)
  }
}
