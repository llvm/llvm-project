// RUN: mlir-translate -mlir-to-cpp %s | FileCheck %s --check-prefix NO-FILTER
// RUN: mlir-translate -mlir-to-cpp -file-id=non-existing %s | FileCheck %s --check-prefix NON-EXISTING
// RUN: mlir-translate -mlir-to-cpp -file-id=file_one %s | FileCheck %s --check-prefix FILE-ONE
// RUN: mlir-translate -mlir-to-cpp -file-id=file_two %s | FileCheck %s --check-prefix FILE-TWO


// NO-FILTER-NOT: func_one
// NO-FILTER-NOT: func_two

// NON-EXISTING-NOT: func_one
// NON-EXISTING-NOT: func_two

// FILE-ONE: func_one
// FILE-ONE-NOT: func_two

// FILE-TWO-NOT: func_one
// FILE-TWO: func_two

emitc.file "file_one" {
  emitc.func @func_one(%arg: f32) {
    emitc.return
  }
}

emitc.file "file_two" {
  emitc.func @func_two(%arg: f32) {
    emitc.return
  }
}
