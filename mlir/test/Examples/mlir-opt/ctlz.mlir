// This file is left in-tree despite having no assertions so it can be
// referenced by the tutorial text.

// RUN: mlir-opt %s

func.func @main(%arg0: i32) -> i32 {
  %0 = math.ctlz %arg0 : i32
  func.return %0 : i32
}
