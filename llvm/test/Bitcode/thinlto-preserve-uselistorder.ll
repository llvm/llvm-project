; Check that thin lto bitcode respects preserve-bc-uselistorder

; RUN: opt --preserve-bc-uselistorder --thinlto-bc --thinlto-split-lto-unit < %s | llvm-dis --preserve-ll-uselistorder | FileCheck %s

; CHECK: uselistorder ptr @g, { 3, 2, 1, 0 }

@g = external global i32

define void @func1() {
  load i32, ptr @g
  load i32, ptr @g
  ret void
}

define void @func2() {
  load i32, ptr @g
  load i32, ptr @g
  ret void
}

uselistorder ptr @g, { 3, 2, 1, 0 }
