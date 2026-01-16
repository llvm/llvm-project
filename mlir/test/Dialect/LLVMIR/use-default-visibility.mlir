// RUN: mlir-opt --llvm-use-default-visibility=visibility=hidden %s | FileCheck %s --check-prefix=HIDDEN
// RUN: mlir-opt --llvm-use-default-visibility=visibility=protected %s | FileCheck %s --check-prefix=PROTECTED

// Ensure the global function definitions and global values are changed to the specified visibility,
// and only when they have default visibility.

llvm.comdat @llvm_comdat {
  llvm.comdat_selector @any any
}

llvm.func @func() {
  llvm.return
}

llvm.func @ifunc_resolver() -> !llvm.ptr {
  %0 = llvm.mlir.addressof @func : !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

// Default visibility

llvm.func @func1() {
  llvm.return
}

llvm.mlir.global internal constant @global1(0 : i32) : i32

llvm.mlir.alias external @func1_alias {addr_space = 0 : i32} : !llvm.ptr {
  %0 = llvm.mlir.addressof @func1 : !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

llvm.func @decl1()

llvm.mlir.global internal constant @comdat1(1 : i64) comdat(@llvm_comdat::@any) : i64

llvm.mlir.ifunc external @ifunc1: !llvm.func<void ()>, !llvm.ptr @ifunc_resolver

// Hidden visibility

llvm.func hidden @func2() {
  llvm.return
}

llvm.mlir.global internal hidden constant @global2(1 : i32) : i32

llvm.mlir.alias external hidden @func2_alias {addr_space = 0 : i32} : !llvm.ptr {
  %0 = llvm.mlir.addressof @func2 : !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

llvm.func hidden @decl2()

llvm.mlir.global internal hidden constant @comdat2(1 : i64) comdat(@llvm_comdat::@any) : i64

llvm.mlir.ifunc external hidden @ifunc2: !llvm.func<void ()>, !llvm.ptr @ifunc_resolver

// Protected visibility

llvm.func protected @func3() {
  llvm.return
}

llvm.mlir.global internal protected constant @global3(2 : i32) : i32

llvm.mlir.alias external protected @func3_alias {addr_space = 0 : i32} : !llvm.ptr {
  %0 = llvm.mlir.addressof @func3 : !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

llvm.func protected @decl3()

llvm.mlir.global internal protected constant @comdat3(1 : i64) comdat(@llvm_comdat::@any) : i64

llvm.mlir.ifunc external protected @ifunc3: !llvm.func<void ()>, !llvm.ptr @ifunc_resolver

// HIDDEN-LABEL:   llvm.comdat @llvm_comdat {
// HIDDEN:           llvm.comdat_selector @any any
// HIDDEN:         }

// HIDDEN-LABEL:   llvm.func hidden @func() {
// HIDDEN:           llvm.return
// HIDDEN:         }

// HIDDEN-LABEL:   llvm.func hidden @ifunc_resolver() -> !llvm.ptr {
// HIDDEN:           %[[MLIR_0:.*]] = llvm.mlir.addressof @func : !llvm.ptr
// HIDDEN:           llvm.return %[[MLIR_0]] : !llvm.ptr
// HIDDEN:         }

// HIDDEN-LABEL:   llvm.func hidden @func1() {
// HIDDEN:           llvm.return
// HIDDEN:         }
// HIDDEN:         llvm.mlir.global internal hidden constant @global1(0 : i32) {addr_space = 0 : i32} : i32

// HIDDEN-LABEL:   llvm.mlir.alias external hidden @func1_alias {addr_space = 0 : i32} : !llvm.ptr {
// HIDDEN:           %[[MLIR_0:.*]] = llvm.mlir.addressof @func1 : !llvm.ptr
// HIDDEN:           llvm.return %[[MLIR_0]] : !llvm.ptr
// HIDDEN:         }
// HIDDEN:         llvm.func hidden @decl1()
// HIDDEN:         llvm.mlir.global internal hidden constant @comdat1(1 : i64) comdat(@llvm_comdat::@any) {addr_space = 0 : i32} : i64
// HIDDEN:         llvm.mlir.ifunc external hidden @ifunc1 : !llvm.func<void ()>, !llvm.ptr @ifunc_resolver

// HIDDEN-LABEL:   llvm.func hidden @func2() {
// HIDDEN:           llvm.return
// HIDDEN:         }
// HIDDEN:         llvm.mlir.global internal hidden constant @global2(1 : i32) {addr_space = 0 : i32} : i32

// HIDDEN-LABEL:   llvm.mlir.alias external hidden @func2_alias {addr_space = 0 : i32} : !llvm.ptr {
// HIDDEN:           %[[MLIR_0:.*]] = llvm.mlir.addressof @func2 : !llvm.ptr
// HIDDEN:           llvm.return %[[MLIR_0]] : !llvm.ptr
// HIDDEN:         }
// HIDDEN:         llvm.func hidden @decl2()
// HIDDEN:         llvm.mlir.global internal hidden constant @comdat2(1 : i64) comdat(@llvm_comdat::@any) {addr_space = 0 : i32} : i64
// HIDDEN:         llvm.mlir.ifunc external hidden @ifunc2 : !llvm.func<void ()>, !llvm.ptr @ifunc_resolver

// HIDDEN-LABEL:   llvm.func protected @func3() {
// HIDDEN:           llvm.return
// HIDDEN:         }
// HIDDEN:         llvm.mlir.global internal protected constant @global3(2 : i32) {addr_space = 0 : i32} : i32

// HIDDEN-LABEL:   llvm.mlir.alias external protected @func3_alias {addr_space = 0 : i32} : !llvm.ptr {
// HIDDEN:           %[[MLIR_0:.*]] = llvm.mlir.addressof @func3 : !llvm.ptr
// HIDDEN:           llvm.return %[[MLIR_0]] : !llvm.ptr
// HIDDEN:         }
// HIDDEN:         llvm.func protected @decl3()
// HIDDEN:         llvm.mlir.global internal protected constant @comdat3(1 : i64) comdat(@llvm_comdat::@any) {addr_space = 0 : i32} : i64
// HIDDEN:         llvm.mlir.ifunc external protected @ifunc3 : !llvm.func<void ()>, !llvm.ptr @ifunc_resolver

// PROTECTED-LABEL:   llvm.comdat @llvm_comdat {
// PROTECTED:           llvm.comdat_selector @any any
// PROTECTED:         }

// PROTECTED-LABEL:   llvm.func protected @func() {
// PROTECTED:           llvm.return
// PROTECTED:         }

// PROTECTED-LABEL:   llvm.func protected @ifunc_resolver() -> !llvm.ptr {
// PROTECTED:           %[[MLIR_0:.*]] = llvm.mlir.addressof @func : !llvm.ptr
// PROTECTED:           llvm.return %[[MLIR_0]] : !llvm.ptr
// PROTECTED:         }

// PROTECTED-LABEL:   llvm.func protected @func1() {
// PROTECTED:           llvm.return
// PROTECTED:         }
// PROTECTED:         llvm.mlir.global internal protected constant @global1(0 : i32) {addr_space = 0 : i32} : i32

// PROTECTED-LABEL:   llvm.mlir.alias external protected @func1_alias {addr_space = 0 : i32} : !llvm.ptr {
// PROTECTED:           %[[MLIR_0:.*]] = llvm.mlir.addressof @func1 : !llvm.ptr
// PROTECTED:           llvm.return %[[MLIR_0]] : !llvm.ptr
// PROTECTED:         }
// PROTECTED:         llvm.func protected @decl1()
// PROTECTED:         llvm.mlir.global internal protected constant @comdat1(1 : i64) comdat(@llvm_comdat::@any) {addr_space = 0 : i32} : i64
// PROTECTED:         llvm.mlir.ifunc external protected @ifunc1 : !llvm.func<void ()>, !llvm.ptr @ifunc_resolver

// PROTECTED-LABEL:   llvm.func hidden @func2() {
// PROTECTED:           llvm.return
// PROTECTED:         }
// PROTECTED:         llvm.mlir.global internal hidden constant @global2(1 : i32) {addr_space = 0 : i32} : i32

// PROTECTED-LABEL:   llvm.mlir.alias external hidden @func2_alias {addr_space = 0 : i32} : !llvm.ptr {
// PROTECTED:           %[[MLIR_0:.*]] = llvm.mlir.addressof @func2 : !llvm.ptr
// PROTECTED:           llvm.return %[[MLIR_0]] : !llvm.ptr
// PROTECTED:         }
// PROTECTED:         llvm.func hidden @decl2()
// PROTECTED:         llvm.mlir.global internal hidden constant @comdat2(1 : i64) comdat(@llvm_comdat::@any) {addr_space = 0 : i32} : i64
// PROTECTED:         llvm.mlir.ifunc external hidden @ifunc2 : !llvm.func<void ()>, !llvm.ptr @ifunc_resolver

// PROTECTED-LABEL:   llvm.func protected @func3() {
// PROTECTED:           llvm.return
// PROTECTED:         }
// PROTECTED:         llvm.mlir.global internal protected constant @global3(2 : i32) {addr_space = 0 : i32} : i32

// PROTECTED-LABEL:   llvm.mlir.alias external protected @func3_alias {addr_space = 0 : i32} : !llvm.ptr {
// PROTECTED:           %[[MLIR_0:.*]] = llvm.mlir.addressof @func3 : !llvm.ptr
// PROTECTED:           llvm.return %[[MLIR_0]] : !llvm.ptr
// PROTECTED:         }
// PROTECTED:         llvm.func protected @decl3()
// PROTECTED:         llvm.mlir.global internal protected constant @comdat3(1 : i64) comdat(@llvm_comdat::@any) {addr_space = 0 : i32} : i64
// PROTECTED:         llvm.mlir.ifunc external protected @ifunc3 : !llvm.func<void ()>, !llvm.ptr @ifunc_resolver
