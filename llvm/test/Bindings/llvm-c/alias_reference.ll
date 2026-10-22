; RUN: llvm-as < %s | llvm-dis > %t.orig
; RUN: llvm-as < %s | llvm-c-test --echo > %t.echo
; RUN: diff -w %t.orig %t.echo
;
; Test using global aliases as constant references.

source_filename = "alias_reference.ll"

@global_var = global i32 100
@global_alias1 = alias i32, ptr @global_var
@global_alias2 = alias i32, ptr @global_alias1

; Use alias as a constant in a global initializer
@ptr_to_alias = global ptr @global_alias1

; Use alias in function
define i32 @load_from_alias() {
  %1 = load i32, ptr @global_alias1
  ret i32 %1
}

define ptr @get_alias_addr() {
  ret ptr @global_alias2
}

; Use alias in constant expression
@const_expr_with_alias = global ptr getelementptr (i32, ptr @global_alias1, i64 0)

; Function that references alias
define i32 @use_multiple_aliases() {
  %1 = load i32, ptr @global_alias1
  %2 = load i32, ptr @global_alias2
  %3 = add i32 %1, %2
  ret i32 %3
}

; External alias
@external_alias = external alias i32, ptr @global_var

; Weak alias
@weak_alias = weak alias i32, ptr @global_var

; Test all aliases are properly cloned
define i32 @test_all_aliases() {
  %1 = load i32, ptr @global_alias1
  %2 = load i32, ptr @global_alias2
  %3 = load i32, ptr @weak_alias
  %4 = add i32 %1, %2
  %5 = add i32 %4, %3
  ret i32 %5
}
