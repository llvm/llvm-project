; RUN: rm -rf %t && split-file %s %t
; RUN: not llvm-link %t/non-variable.ll %t/non-variable.ll -S -o - 2>&1 | FileCheck %s --check-prefix=NONVARIABLE
; RUN: not llvm-link %t/no-base-object.ll %t/no-base-object-aux.ll -S -o - 2>&1 | FileCheck %s --check-prefix=NOSIZE

;--- non-variable.ll
; NONVARIABLE: GlobalVariable required for data dependent selection!
$c1 = comdat largest

define void @c1() comdat($c1) {
  ret void
}

;--- no-base-object.ll
; NOSIZE: COMDAT key involves incomputable alias size.
$c1 = comdat largest

@some_name = unnamed_addr constant i32 42, comdat($c1)
@c1 = alias i8, inttoptr (i32 1 to ptr)

;--- no-base-object-aux.ll
$c1 = comdat largest

@some_name = private unnamed_addr constant i32 42, comdat($c1)
@c1 = alias i32, ptr @some_name
