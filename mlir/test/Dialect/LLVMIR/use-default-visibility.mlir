// RUN: mlir-opt --llvm-use-default-visibility=visibility=hidden %s | FileCheck %s --check-prefix=HIDDEN
// RUN: mlir-opt --llvm-use-default-visibility=visibility=protected %s | FileCheck %s --check-prefix=PROTECTED

// Ensure the global function definitions and global values are changed to the specified visibility,
// and only when they have default visibility.

llvm.func @func1() {
  llvm.return
}

llvm.mlir.global internal constant @global1(0 : i32) : i32

llvm.func hidden @func2() {
  llvm.return
}

llvm.mlir.global internal hidden constant @global2(1 : i32) : i32

llvm.func protected @func3() {
  llvm.return
}

llvm.mlir.global internal protected constant @global3(2 : i32) : i32

llvm.comdat @llvm_comdat {
  llvm.comdat_selector @any any
}
llvm.mlir.global internal constant @global_comdat(1 : i64) comdat(@llvm_comdat::@any) : i64

llvm.func @decl()

// HIDDEN-LABEL:   llvm.func hidden @func1() {
// HIDDEN:           llvm.return
// HIDDEN:         }
// HIDDEN:         llvm.mlir.global internal hidden constant @global1(0 : i32) {addr_space = 0 : i32} : i32

// HIDDEN-LABEL:   llvm.func hidden @func2() {
// HIDDEN:           llvm.return
// HIDDEN:         }
// HIDDEN:         llvm.mlir.global internal hidden constant @global2(1 : i32) {addr_space = 0 : i32} : i32

// HIDDEN-LABEL:   llvm.func protected @func3() {
// HIDDEN:           llvm.return
// HIDDEN:         }
// HIDDEN:         llvm.mlir.global internal protected constant @global3(2 : i32) {addr_space = 0 : i32} : i32

// HIDDEN-LABEL:   llvm.comdat @llvm_comdat {
// HIDDEN:           llvm.comdat_selector @any any
// HIDDEN:         }
// HIDDEN:         llvm.mlir.global internal hidden constant @global_comdat(1 : i64) comdat(@llvm_comdat::@any) {addr_space = 0 : i32} : i64
// HIDDEN:         llvm.func hidden @decl()

// PROTECTED-LABEL:   llvm.func protected @func1() {
// PROTECTED:           llvm.return
// PROTECTED:         }
// PROTECTED:         llvm.mlir.global internal protected constant @global1(0 : i32) {addr_space = 0 : i32} : i32

// PROTECTED-LABEL:   llvm.func hidden @func2() {
// PROTECTED:           llvm.return
// PROTECTED:         }
// PROTECTED:         llvm.mlir.global internal hidden constant @global2(1 : i32) {addr_space = 0 : i32} : i32

// PROTECTED-LABEL:   llvm.func protected @func3() {
// PROTECTED:           llvm.return
// PROTECTED:         }
// PROTECTED:         llvm.mlir.global internal protected constant @global3(2 : i32) {addr_space = 0 : i32} : i32

// PROTECTED-LABEL:   llvm.comdat @llvm_comdat {
// PROTECTED:           llvm.comdat_selector @any any
// PROTECTED:         }
// PROTECTED:         llvm.mlir.global internal protected constant @global_comdat(1 : i64) comdat(@llvm_comdat::@any) {addr_space = 0 : i32} : i64
// PROTECTED:         llvm.func protected @decl()
