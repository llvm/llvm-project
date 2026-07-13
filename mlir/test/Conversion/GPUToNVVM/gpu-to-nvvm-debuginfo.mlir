// RUN: mlir-opt %s -convert-gpu-to-nvvm='has-redux=1' -mlir-print-debuginfo | FileCheck %s

#di_file = #llvm.di_file<"foo.mlir" in "/tmp/">
#di_compile_unit = #llvm.di_compile_unit<
  id = distinct[0]<>, sourceLanguage = DW_LANG_C, file = #di_file,
  producer = "MLIR", isOptimized = true, emissionKind = Full
>
#di_subprogram = #llvm.di_subprogram<
  compileUnit = #di_compile_unit, scope = #di_file, name = "test_const_printf_with_loc",
  file = #di_file, subprogramFlags = "Definition"
>

// CHECK-DAG: [[LOC:#[a-zA-Z0-9_]+]] = loc("foo.mlir":0:0)
#loc = loc("foo.mlir":0:0)

// Check that debug info metadata from the function is removed from the global location.
gpu.module @test_module_1 {
  // CHECK-DAG: llvm.mlir.global internal constant @[[$PRINT_GLOBAL0:[A-Za-z0-9_]+]]("Hello, world with location\0A\00") {addr_space = 0 : i32} loc([[LOC]])
  // CHECK-DAG: llvm.func @vprintf(!llvm.ptr, !llvm.ptr) -> i32 loc([[LOC]])

  gpu.func @test_const_printf_with_loc() {
    gpu.printf "Hello, world with location\n" loc(fused<#di_subprogram>[#loc])
    gpu.return
  }
}

// Check that debug info metadata from the function is removed from the global location.
gpu.module @test_module_2 {
  // CHECK-DAG: llvm.func @__nv_abs(i32) -> i32 loc([[LOC]])
  func.func @gpu_abs_with_loc(%arg_i32 : i32) -> (i32) {
    %result32 = math.absi %arg_i32 : i32 loc(fused<#di_subprogram>[#loc])
    func.return %result32 : i32
  }
}
