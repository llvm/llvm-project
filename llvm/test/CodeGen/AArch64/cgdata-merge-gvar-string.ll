; This test verifies that global variables (string) are hashed based on their initial contents,
; allowing them to be merged even if they appear different due to their names.
; Now they become identical functions that can be merged without creating a parameter.

; RUN: llc -mtriple=arm64-apple-darwin -enable-global-merge-func=true -global-merging-skip-no-params=false < %s | FileCheck %s

; CHECK: _f1.Tgm
; CHECK: _f2.Tgm
; CHECK-NOT: _f3.Tgm
; CHECK-NOT: _f4.Tgm

; The initial contents of `.str` and `.str.1` are identical, but not with those of `.str.2` and `.str.3`.
@.str = private unnamed_addr constant [6 x i8] c"hello\00", align 1
@.str.1 = private unnamed_addr constant [6 x i8] c"hello\00", align 1
@.str.2 = private unnamed_addr constant [6 x i8] c"diff2\00", align 1
@.str.3 = private unnamed_addr constant [6 x i8] c"diff3\00", align 1

declare i32 @goo(ptr noundef)

define i32 @f1() {
entry:
  %call = tail call i32 @goo(ptr noundef nonnull @.str)
  %add = add nsw i32 %call, 1
  ret i32 %add
}

define i32 @f2() {
entry:
  %call = tail call i32 @goo(ptr noundef nonnull @.str.1)
  %add = add nsw i32 %call, 1
  ret i32 %add
}

define i32 @f3() {
entry:
  %call = tail call noundef i32 @goo(ptr noundef nonnull @.str.2)
  %add = sub nsw i32 %call, 1
  ret i32 %add
}

define i32 @f4() {
entry:
  %call = tail call noundef i32 @goo(ptr noundef nonnull @.str.3)
  %add = sub nsw i32 %call, 1
  ret i32 %add
}
