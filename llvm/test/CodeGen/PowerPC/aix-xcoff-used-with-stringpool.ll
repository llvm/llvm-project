;; Test that the string pooling pass does not pool globals that are
;; in llvm.used or in llvm.compiler.used.

; RUN: llc -verify-machineinstrs -mcpu=pwr8 -mtriple powerpc-ibm-aix-xcoff -data-sections=false < %s | \
; RUN:   FileCheck %s

; RUN: llc -verify-machineinstrs -mcpu=pwr8 -mtriple powerpc64-ibm-aix-xcoff -data-sections=false < %s | \
; RUN:   FileCheck %s

@keep_this = internal constant [5 x i8] c"keep1", align 1
@keep_this2 = internal constant [5 x i8] c"keep2", align 1
@.str.1 = private unnamed_addr constant [12 x i8] c"str1_STRING\00", align 1
@.str.2 = private unnamed_addr constant [12 x i8] c"str2_STRING\00", align 1
@.str.3 = private unnamed_addr constant [12 x i8] c"str3_STRING\00", align 1
@llvm.used = appending global [1 x ptr] [ptr @keep_this], section "llvm.metadata"
@llvm.compiler.used = appending global [1 x ptr] [ptr @keep_this2], section "llvm.metadata"

declare signext i32 @callee(ptr noundef)

define dso_local signext i32 @keep1() {
entry:
  %call = tail call signext i32 @callee(ptr noundef nonnull @keep_this)
  ret i32 %call
}

define dso_local signext i32 @keep2() {
entry:
  %call = tail call signext i32 @callee(ptr noundef nonnull @keep_this2)
  ret i32 %call
}

define dso_local signext i32 @str1() {
entry:
  %call = tail call signext i32 @callee(ptr noundef nonnull @.str.1)
  ret i32 %call
}

define dso_local signext i32 @str2() {
entry:
  %call = tail call signext i32 @callee(ptr noundef nonnull @.str.2)
  ret i32 %call
}

define dso_local signext i32 @str3() {
entry:
  %call = tail call signext i32 @callee(ptr noundef nonnull @.str.3)
  ret i32 %call
}

; CHECK:    .lglobl keep_this
; CHECK:  keep_this:
; CHECK:    .lglobl keep_this2
; CHECK:  keep_this2:
; CHECK:  L..__ModuleStringPool:
; CHECK:    .string "str1_STRING"
; CHECK:    .string "str2_STRING"
; CHECK:    .string "str3_STRING"
