// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// Per-element string globals referenced by the array-of-pointers global below.
llvm.mlir.global private constant @s0("a\00") {addr_space = 0 : i32} : !llvm.array<2 x i8>
llvm.mlir.global private constant @s1("b\00") {addr_space = 0 : i32} : !llvm.array<2 x i8>
llvm.mlir.global private constant @s2("c\00") {addr_space = 0 : i32} : !llvm.array<2 x i8>

// Single llvm.mlir.constant with a FlatSymbolRefAttr leaf per array element,
// emitted inside the global's initializer region.  This shape lets MLIR's
// translator route the lowering through llvm::ConstantArray::get once
// (the existing array-of-ArrayAttr path in getLLVMConstant), avoiding the
// O(N^2) per-element ConstantFoldInsertValueInstruction blowup that a chain
// of llvm.insertvalue ops would otherwise force on large arrays.
llvm.mlir.global external constant @ptrs() : !llvm.array<3 x ptr> {
  %0 = llvm.mlir.constant([@s0, @s1, @s2]) : !llvm.array<3 x ptr>
  llvm.return %0 : !llvm.array<3 x ptr>
}

// CHECK: @ptrs = constant [3 x ptr] [ptr @s0, ptr @s1, ptr @s2]
