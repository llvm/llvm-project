// RUN: mlir-opt %s -finalize-memref-to-llvm -split-input-file -verify-diagnostics

// expected-error@+1{{redefinition of reserved function 'malloc' of different type '!llvm.func<void (i64)>' is prohibited}}
llvm.func @malloc(i64)
func.func @redef_reserved() {
  %alloc = memref.alloc() : memref<1024x64xf32, 1>
  llvm.return
}
