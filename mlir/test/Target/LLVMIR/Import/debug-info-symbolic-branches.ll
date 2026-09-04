; RUN: mlir-translate -import-llvm -mlir-print-debuginfo -emit-expensive-warnings %s 2>&1 | FileCheck %s

; Import a forward branch and a backward skip, and make sure MLIR keeps the op
; order and label IDs.

; CHECK: llvm.intr.dbg.value {{.*}} #llvm.di_expression<[DW_OP_LLVM_label(0), DW_OP_LLVM_bra(42), DW_OP_LLVM_skip(0), DW_OP_LLVM_label(42)]> = {{.*}} : i64
define void @f(i64 %x) !dbg !4 {
  #dbg_value(i64 %x, !DILocalVariable(scope: !4),
             !DIExpression(DW_OP_LLVM_label, 0, DW_OP_LLVM_bra, 42,
                           DW_OP_LLVM_skip, 0, DW_OP_LLVM_label, 42),
             !DILocation(scope: !4))
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1)
!1 = !DIFile(filename: "test.c", directory: "/")
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "f", scope: !1,
                            type: !DISubroutineType(types: !{null}),
                            spFlags: DISPFlagDefinition, unit: !0)
