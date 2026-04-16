// RUN: mlir-opt -convert-scf-to-spirv %s | FileCheck %s

// CHECK-LABEL:   spirv.func @main() "None" {
// CHECK:           %[[VAL_0:.*]] = spirv.Constant -5 : si32
// CHECK:           %[[FROM_ELEMENTS_0:.*]] = vector.from_elements %[[VAL_0]] : vector<1xsi32>
// CHECK:           spirv.Return
func.func @main() {
  %2 = spirv.Constant -5 : si32
  %3 = vector.from_elements %2 : vector<1xsi32>
  return
}

