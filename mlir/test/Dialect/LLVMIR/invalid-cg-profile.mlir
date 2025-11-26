// RUN: mlir-translate %s -mlir-to-llvmir | FileCheck %s
// CHECK: !llvm.module.flags = !{!0, !3}
// CHECK: !0 = !{i32 5, !"CG Profile", !1}
// CHECK: !1 = distinct !{!2, !2, !2}
// CHECK: !2 = !{null, null, i64 222}
// CHECK: !3 = !{i32 2, !"Debug Info Version", i32 3}

module {
  llvm.module_flags [#llvm.mlir.module_flag<append, "CG Profile", [
                       #llvm.cgprofile_entry<from = @from, to = @to, count = 222>,
                       #llvm.cgprofile_entry<from = @from, count = 222>,
                       #llvm.cgprofile_entry<from = @to, to = @from, count = 222>
                    ]>]
}
