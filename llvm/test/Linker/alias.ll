; RUN: llvm-link %s %S/Inputs/alias.ll -S -o - | FileCheck --check-prefix=C1 %s
; RUN: llvm-link %S/Inputs/alias.ll %s -S -o - | FileCheck --check-prefix=C2 %s

; FIXME:
; The C1 direction is incorrect.
; When moving an alias to an existing module and we want to discard the aliasee
; (the C2 case), the IRMover knows to copy the aliasee as internal.
; When moving a replacement to an aliasee to a module that has an alias (C1),
; a replace all uses with blindly changes the alias.
; The C1 case doesn't happen when using a system linker with a plugin because
; the linker does full symbol resolution first.
; Given that this is a problem only with llvm-link and its 1 module at a time
; linking, it should probably learn to changes the aliases in the destination
; before using the IRMover.

@foo = weak global i32 0
; C1-DAG: @foo = alias i32, ptr @zed
; C2-DAG: @foo = alias i32, ptr @zed

@bar = alias i32, ptr @foo
; C1-DAG: @bar = alias i32, ptr @foo

; C2-DAG: @foo.1 = internal global i32 0
; C2-DAG: @bar = alias i32, ptr @foo.1

@foo2 = weak global i32 0
; C1-DAG: @foo2 = alias i16, ptr @zed
; C2-DAG: @foo2 = alias i16, ptr @zed

@bar2 = alias i32, ptr @foo2
; C1-DAG: @bar2 = alias i32, ptr @foo2

; C2-DAG: @foo2.2 = internal global i32 0
; C2-DAG: @bar2 = alias i32, ptr @foo2.2

; C1-DAG: @zed = global i32 42
; C2-DAG: @zed = global i32 42
