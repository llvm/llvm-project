// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// CHECK: irdl.dialect @testvar {
irdl.dialect @testvar {

  // CHECK-LABEL: irdl.operation @single_operand {
  // CHECK-NEXT:    %[[v0:[^ ]*]] = irdl.is i32
  // CHECK-NEXT:    irdl.operands(%[[v0]])
  // CHECK-NEXT:  }
  irdl.operation @single_operand {
    %0 = irdl.is i32
    irdl.operands(single %0)
  }

  // CHECK-LABEL: irdl.operation @var_operand {
  // CHECK-NEXT:    %[[v0:[^ ]*]] = irdl.is i16 
  // CHECK-NEXT:    %[[v1:[^ ]*]] = irdl.is i32 
  // CHECK-NEXT:    %[[v2:[^ ]*]] = irdl.is i64 
  // CHECK-NEXT:    irdl.operands(%[[v0]], variadic %[[v1]], %[[v2]])
  // CHECK-NEXT:  }
  irdl.operation @var_operand {
    %0 = irdl.is i16
    %1 = irdl.is i32
    %2 = irdl.is i64
    irdl.operands(%0, variadic %1, %2)
  }

  // CHECK-LABEL: irdl.operation @opt_operand {
  // CHECK-NEXT:    %[[v0:[^ ]*]] = irdl.is i16 
  // CHECK-NEXT:    %[[v1:[^ ]*]] = irdl.is i32 
  // CHECK-NEXT:    %[[v2:[^ ]*]] = irdl.is i64 
  // CHECK-NEXT:    irdl.operands(%[[v0]], optional %[[v1]], %[[v2]])
  // CHECK-NEXT:  }
  irdl.operation @opt_operand {
    %0 = irdl.is i16
    %1 = irdl.is i32
    %2 = irdl.is i64
    irdl.operands(%0, optional %1, %2)
  }

  // CHECK-LABEL: irdl.operation @var_and_opt_operand {
  // CHECK-NEXT:    %[[v0:[^ ]*]] = irdl.is i16 
  // CHECK-NEXT:    %[[v1:[^ ]*]] = irdl.is i32 
  // CHECK-NEXT:    %[[v2:[^ ]*]] = irdl.is i64 
  // CHECK-NEXT:    irdl.operands(variadic %[[v0]], optional %[[v1]], %[[v2]])
  // CHECK-NEXT:  }
  irdl.operation @var_and_opt_operand {
    %0 = irdl.is i16
    %1 = irdl.is i32
    %2 = irdl.is i64
    irdl.operands(variadic %0, optional %1, %2)
  }


  // CHECK-LABEL: irdl.operation @single_result {
  // CHECK-NEXT:    %[[v0:[^ ]*]] = irdl.is i32
  // CHECK-NEXT:    irdl.results(%[[v0]])
  // CHECK-NEXT:  }
  irdl.operation @single_result {
    %0 = irdl.is i32
    irdl.results(single %0)
  }

  // CHECK-LABEL: irdl.operation @var_result {
  // CHECK-NEXT:    %[[v0:[^ ]*]] = irdl.is i16 
  // CHECK-NEXT:    %[[v1:[^ ]*]] = irdl.is i32 
  // CHECK-NEXT:    %[[v2:[^ ]*]] = irdl.is i64 
  // CHECK-NEXT:    irdl.results(%[[v0]], variadic %[[v1]], %[[v2]])
  // CHECK-NEXT:  }
  irdl.operation @var_result {
    %0 = irdl.is i16
    %1 = irdl.is i32
    %2 = irdl.is i64
    irdl.results(%0, variadic %1, %2)
  }

  // CHECK-LABEL: irdl.operation @opt_result {
  // CHECK-NEXT:    %[[v0:[^ ]*]] = irdl.is i16 
  // CHECK-NEXT:    %[[v1:[^ ]*]] = irdl.is i32 
  // CHECK-NEXT:    %[[v2:[^ ]*]] = irdl.is i64 
  // CHECK-NEXT:    irdl.results(%[[v0]], optional %[[v1]], %[[v2]])
  // CHECK-NEXT:  }
  irdl.operation @opt_result {
    %0 = irdl.is i16
    %1 = irdl.is i32
    %2 = irdl.is i64
    irdl.results(%0, optional %1, %2)
  }

  // CHECK-LABEL: irdl.operation @var_and_opt_result {
  // CHECK-NEXT:    %[[v0:[^ ]*]] = irdl.is i16 
  // CHECK-NEXT:    %[[v1:[^ ]*]] = irdl.is i32 
  // CHECK-NEXT:    %[[v2:[^ ]*]] = irdl.is i64 
  // CHECK-NEXT:    irdl.results(variadic %[[v0]], optional %[[v1]], %[[v2]])
  // CHECK-NEXT:  }
  irdl.operation @var_and_opt_result {
    %0 = irdl.is i16
    %1 = irdl.is i32
    %2 = irdl.is i64
    irdl.results(variadic %0, optional %1, %2)
  }
}
