// RUN: mlir-opt %s -inline -mlir-print-debuginfo | FileCheck %s

llvm.func @foo() -> !llvm.ptr

llvm.func @with_byval_arg(%ptr : !llvm.ptr { llvm.byval = f64 }) {
  llvm.return
}

// CHECK-LABEL: llvm.func @test_byval
llvm.func @test_byval() {
  // CHECK: %[[COPY:.+]] = llvm.alloca %{{.+}} x f64
  // CHECK-SAME: loc(#[[$LOC:.+]])
  // CHECK: %[[ORIG:.+]] = llvm.call @foo() : () -> !llvm.ptr loc(#[[$LOC]])
  %0 = llvm.call @foo() : () -> !llvm.ptr loc("inlining-debuginfo.mlir":14:2)
  // CHECK: "llvm.intr.memcpy"(%[[COPY]], %[[ORIG]]
  // CHECK-SAME: loc(#[[$LOC]])
  llvm.call @with_byval_arg(%0) : (!llvm.ptr) -> ()
  llvm.return
}

llvm.func @leaf() attributes {no_inline} {
  llvm.return
} loc(#leaf_loc)

llvm.func @wrapper() {
  llvm.call @leaf() : () -> () loc(unknown)
  llvm.return
} loc(unknown)

// CHECK-LABEL: llvm.func @caller()
// CHECK: llvm.call @leaf() : () -> () loc(#[[INLINED:.+]])
llvm.func @caller() {
  llvm.call @wrapper() : () -> () loc(unknown)
  llvm.return
} loc(#caller_loc)

#file = #llvm.di_file<"test.c" in "/">
#cu = #llvm.di_compile_unit<id = distinct[0]<>, sourceLanguage = DW_LANG_C, file = #file, isOptimized = true>
#caller = #llvm.di_subprogram<id = distinct[1]<>, compileUnit = #cu, scope = #file, name = "caller", file = #file, subprogramFlags = Definition, type = #llvm.di_subroutine_type<>>
#leaf = #llvm.di_subprogram<id = distinct[2]<>, compileUnit = #cu, scope = #file, name = "leaf", file = #file, subprogramFlags = Definition, type = #llvm.di_subroutine_type<>>
#caller_loc = loc(fused<#caller>["test.c":10:1])
#leaf_loc = loc(fused<#leaf>["test.c":1:1])

// CHECK: #[[$LOC]] = loc("inlining-debuginfo.mlir":14:2)
// CHECK: #[[UNKNOWN:.+]] = loc(unknown)
// CHECK: #[[CALLSITE:.+]] = loc("-":0:0)
// CHECK: #[[INLINED]] = loc(callsite(#[[UNKNOWN]] at #[[CALLSITE]]))
