// RUN: mlir-translate %s -mlir-to-llvmir | FileCheck %s
// CHECK: !llvm.module.flags

module {
  llvm.module_flags [#llvm.mlir.module_flag<append, "CG Profile", [
                       #llvm.cgprofile_entry<from = @from, to = @to, count = 222>,
                       #llvm.cgprofile_entry<from = @from, count = 222>,
                       #llvm.cgprofile_entry<from = @to, to = @from, count = 222>
                    ]>]
}
