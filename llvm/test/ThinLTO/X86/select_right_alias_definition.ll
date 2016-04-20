; RUN: opt -module-summary %s -o %t_main.bc
; RUN: opt -module-summary %p/Inputs/select_right_alias_definition1.ll -o %t1.bc
; RUN: opt -module-summary %p/Inputs/select_right_alias_definition2.ll -o %t2.bc

; Make sure that we always select the right definition for alia foo, whatever
; order the files are linked in.

; Try with one order
; RUN: llvm-lto -thinlto-action=thinlink -o %t.index1.bc %t_main.bc %t1.bc %t2.bc
; RUN: llvm-lto -thinlto-action=import -thinlto-index %t.index1.bc %t_main.bc -o - | llvm-dis -o - | FileCheck %s --check-prefix=IMPORT

; Try with the other order (reversing %t1.bc and %t2.bc)
; RUN: llvm-lto -thinlto-action=thinlink -o %t.index2.bc %t_main.bc %t2.bc %t1.bc
; RUN: llvm-lto -thinlto-action=import -thinlto-index %t.index2.bc %t_main.bc -o - | llvm-dis -o - | FileCheck %s --check-prefix=IMPORT

; IMPORT: @foo = alias i32 (...), bitcast (i32 ()* @foo2 to i32 (...)*)
; IMPORT: define linkonce_odr i32 @foo2() {
; IMPORT-NEXT:  %ret = add i32 42, 42
; IMPORT-NEXT:  ret i32 %ret
; IMPORT-NEXT: }

declare i32 @foo()

define i32 @main() {
    %ret = call i32 @foo()
    ret i32 %ret
}