// RUN: mlir-opt -inline %s | FileCheck %s

func.func private @pattern_body() -> (!pdl.type, !pdl.type, !pdl.operation) {
  %0 = pdl.type : i32
  %1 = pdl.type
  %2 = pdl.operation  -> (%0, %1 : !pdl.type, !pdl.type)
  return %0, %1, %2 : !pdl.type, !pdl.type, !pdl.operation
}

func.func private @rewrite_body(%arg0: !pdl.type, %arg1: !pdl.type, %arg2: !pdl.operation) {
  %0 = pdl.operation "foo.op"  -> (%arg0, %arg1 : !pdl.type, !pdl.type)
  pdl.apply_native_rewrite "NativeRewrite"(%0, %arg2 : !pdl.operation, !pdl.operation)
  return
}

// CHECK-LABEL:   pdl.pattern @nonmaterializable_pattern : benefit(1) nonmaterializable {
// CHECK:           %[[VAL_0:.*]] = type : i32
// CHECK:           %[[VAL_1:.*]] = type
// CHECK:           %[[VAL_2:.*]] = operation  -> (%[[VAL_0]], %[[VAL_1]] : !pdl.type, !pdl.type)
// CHECK:           rewrite %[[VAL_2]] {
// CHECK:             %[[VAL_3:.*]] = operation "foo.op"  -> (%[[VAL_0]], %[[VAL_1]] : !pdl.type, !pdl.type)
// CHECK:             apply_native_rewrite "NativeRewrite"(%[[VAL_3]], %[[VAL_2]] : !pdl.operation, !pdl.operation)
// CHECK:           }
// CHECK:         }
pdl.pattern @nonmaterializable_pattern : benefit(1) nonmaterializable {
  %0:3 = func.call @pattern_body() : () -> (!pdl.type, !pdl.type, !pdl.operation)
  rewrite %0#2 {
    func.call @rewrite_body(%0#0, %0#1, %0#2) : (!pdl.type, !pdl.type, !pdl.operation) -> ()
  }
}
