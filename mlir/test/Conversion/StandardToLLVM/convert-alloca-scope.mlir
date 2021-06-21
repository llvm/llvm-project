// RUN: mlir-opt -convert-std-to-llvm %s | FileCheck %s

// CHECK-LABEL: llvm.func @empty
func @empty() {
  // CHECK: llvm.intr.stacksave 
  // CHECK: llvm.br
  memref.alloca_scope {
    memref.alloca_scope.return
  }
  // CHECK: llvm.intr.stackrestore 
  // CHECK: llvm.br
  // CHECK: llvm.return
  return
}

// CHECK-LABEL: llvm.func @returns_nothing
func @returns_nothing(%b: f32) {
  %a = constant 10.0 : f32
  // CHECK: llvm.intr.stacksave 
  memref.alloca_scope {
    %c = std.addf %a, %b : f32
    memref.alloca_scope.return
  }
  // CHECK: llvm.intr.stackrestore 
  // CHECK: llvm.return
  return
}

// CHECK-LABEL: llvm.func @returns_one_value
func @returns_one_value(%b: f32) -> f32 {
  %a = constant 10.0 : f32
  // CHECK: llvm.intr.stacksave 
  %result = memref.alloca_scope -> f32 {
    %c = std.addf %a, %b : f32
    memref.alloca_scope.return %c: f32
  }
  // CHECK: llvm.intr.stackrestore 
  // CHECK: llvm.return
  return %result : f32
}

// CHECK-LABEL: llvm.func @returns_multiple_values
func @returns_multiple_values(%b: f32) -> f32 {
  %a = constant 10.0 : f32
  // CHECK: llvm.intr.stacksave 
  %result1, %result2 = memref.alloca_scope -> (f32, f32) {
    %c = std.addf %a, %b : f32
    %d = std.subf %a, %b : f32
    memref.alloca_scope.return %c, %d: f32, f32
  }
  // CHECK: llvm.intr.stackrestore 
  // CHECK: llvm.return
  %result = std.addf %result1, %result2 : f32
  return %result : f32
}
