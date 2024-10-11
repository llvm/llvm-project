// RUN: mlir-translate -mlir-to-cpp %s | FileCheck %s

emitc.func @lvalue_variables(%v1: i32, %v2: i32) -> i32 {
  %val = emitc.mul %v1, %v2 : (i32, i32) -> i32
  %variable = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<i32> 
  emitc.assign %val : i32 to %variable : !emitc.lvalue<i32>
  %addr = emitc.apply "&"(%variable) : (!emitc.lvalue<i32>) -> !emitc.ptr<i32>
  emitc.call @zero (%addr) : (!emitc.ptr<i32>) -> ()
  %updated_val = emitc.load %variable : !emitc.lvalue<i32>
  %neg_one = "emitc.constant"() {value = -1 : i32} : () -> i32
  emitc.assign %neg_one : i32 to %variable : !emitc.lvalue<i32>
  emitc.return %updated_val : i32
}
// CHECK-LABEL: int32_t lvalue_variables(
// CHECK-SAME: int32_t [[V1:[^ ]*]], int32_t [[V2:[^ ]*]])
// CHECK-NEXT: int32_t [[VAL:[^ ]*]] = [[V1]] * [[V2]];
// CHECK-NEXT: int32_t [[VAR:[^ ]*]];
// CHECK-NEXT: [[VAR]] = [[VAL]];
// CHECK-NEXT: int32_t* [[VAR_PTR:[^ ]*]] = &[[VAR]];
// CHECK-NEXT: zero([[VAR_PTR]]);
// CHECK-NEXT: int32_t [[VAR_LOAD:[^ ]*]] = [[VAR]]; 
// CHECK-NEXT: int32_t [[NEG_ONE:[^ ]*]] = -1; 
// CHECK-NEXT: [[VAR]] = [[NEG_ONE]];
// CHECK-NEXT: return [[VAR_LOAD]];


emitc.func @zero(%arg0: !emitc.ptr<i32>) attributes {specifiers = ["extern"]}
// CHECK-LABEL: extern void zero(int32_t*);
