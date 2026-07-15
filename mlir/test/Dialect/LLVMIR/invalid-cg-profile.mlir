// RUN: mlir-translate %s -mlir-to-llvmir | FileCheck %s
// CHECK: !llvm.module.flags = !{![[CG_FLAG:[0-9]+]], ![[DBG_FLAG:[0-9]+]]}
// CHECK: ![[CG_FLAG]] = !{i32 5, !"CG Profile", ![[CG_LIST:[0-9]+]]}
// CHECK: ![[CG_LIST]] = distinct !{![[CG_ENTRY:[0-9]+]], ![[CG_ENTRY]], ![[CG_ENTRY]]}
// CHECK: ![[CG_ENTRY]] = !{null, null, i64 222}
// CHECK: ![[DBG_FLAG]] = !{i32 2, !"Debug Info Version", i32 3}

module {
  llvm.module_flags [#llvm.mlir.module_flag<append, "CG Profile", [
                       #llvm.cgprofile_entry<from = @from, to = @to, count = 222>,
                       #llvm.cgprofile_entry<from = @from, count = 222>,
                       #llvm.cgprofile_entry<from = @to, to = @from, count = 222>
                    ]>]
}
