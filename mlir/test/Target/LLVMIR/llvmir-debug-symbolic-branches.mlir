// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// Export a forward branch and a backward skip, and make sure LLVM IR keeps the
// op order and label IDs.

#file = #llvm.di_file<"test.c" in "/">
#cu = #llvm.di_compile_unit<id = distinct[0]<>, sourceLanguage = DW_LANG_C,
                            file = #file>
#sp = #llvm.di_subprogram<compileUnit = #cu, scope = #file, name = "f",
                          subprogramFlags = "Definition",
                          type = #llvm.di_subroutine_type<types = #llvm.di_null_type>>

// CHECK: #dbg_value(i64 %{{.*}}, !{{.*}}, !DIExpression(DW_OP_LLVM_label, 0, DW_OP_LLVM_bra, 42, DW_OP_LLVM_skip, 0, DW_OP_LLVM_label, 42), !{{.*}})
llvm.func @f(%arg: i64) {
  llvm.intr.dbg.value #llvm.di_local_variable<scope = #sp> #llvm.di_expression<[
    DW_OP_LLVM_label(0), DW_OP_LLVM_bra(42), DW_OP_LLVM_skip(0),
    DW_OP_LLVM_label(42)]> = %arg : i64
  llvm.return
} loc(fused<#sp>["test.c":1:1])
