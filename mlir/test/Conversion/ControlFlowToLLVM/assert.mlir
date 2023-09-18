// RUN: mlir-opt %s -convert-cf-to-llvm='use-opaque-pointers=1' | FileCheck %s

// Same below, but using the `ConvertToLLVMPatternInterface` entry point
// and the generic `convert-to-llvm` pass.
// RUN: mlir-opt --convert-to-llvm="filter-dialects=cf" --split-input-file %s | FileCheck %s

func.func @main() {
  %a = arith.constant 0 : i1
  cf.assert %a, "assertion foo"
  return
}

// CHECK: llvm.func @puts(!llvm.ptr)

// CHECK-LABEL: @main
// CHECK: llvm.cond_br %{{.*}}, ^{{.*}}, ^[[FALSE_BRANCH:[[:alnum:]]+]]

// CHECK: ^[[FALSE_BRANCH]]:
// CHECK: %[[ADDRESS_OF:.*]] = llvm.mlir.addressof @{{.*}} : !llvm.ptr{{$}}
// CHECK: %[[GEP:.*]] = llvm.getelementptr %[[ADDRESS_OF]][0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<{{[0-9]+}} x i8>
// CHECK: llvm.call @puts(%[[GEP]]) : (!llvm.ptr) -> ()
