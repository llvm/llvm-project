; RUN: mlir-translate -test-import-llvmir %s | FileCheck %s

; CHECK-LABEL: @custom_load
; CHECK-SAME:  %[[PTR:[[:alnum:]]+]]
define double @custom_load(ptr %ptr) {
  ; CHECK:  %[[LOAD:[0-9]+]] = llvm.load %[[PTR]] : !llvm.ptr -> f64
  ; CHECK:  %[[TEST:[0-9]+]] = "test.same_operand_element_type"(%[[LOAD]], %[[LOAD]]) : (f64, f64) -> f64
  %1 = load double, ptr %ptr
  ; CHECK:   llvm.return %[[TEST]] : f64
  ret double %1
}
