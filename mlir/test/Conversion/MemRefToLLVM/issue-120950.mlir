// RUN: mlir-opt %s -finalize-memref-to-llvm 2>&1 | FileCheck %s

#map = affine_map<(d0) -> (d0 + 1)>
module {
  // CHECK: redefinition of reserved function 'malloc' of different type '!llvm.func<void (i64)>' is prohibited
  llvm.func @malloc(i64)
  func.func @issue_120950() {
    %alloc = memref.alloc() : memref<1024x64xf32, 1>
    llvm.return
  }
}
