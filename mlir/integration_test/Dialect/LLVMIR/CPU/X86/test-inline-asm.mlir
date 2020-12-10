// RUN: mlir-cpu-runner %s -e entry -entry-point-result=void  \
// RUN: -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

module {
  llvm.func @printI64(!llvm.i64)
  llvm.func @entry()  {
    %c2 = llvm.mlir.constant(-42: i64) :!llvm.i64
    %val = llvm.inline_asm "xor $0, $0", "=r,r" %c2 :
      (!llvm.i64) -> !llvm.i64

    // CHECK: 0
    llvm.call @printI64(%val) : (!llvm.i64) -> ()
    llvm.return
  }
}
