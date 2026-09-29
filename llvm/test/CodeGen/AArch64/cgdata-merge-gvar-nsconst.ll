; This test verifies that global variables (ns constant) are hashed based on their initial contents,
; allowing them to be merged even if they appear different due to their names.
; Now they become identical functions that can be merged without creating a parameter

; RUN: llc -mtriple=arm64-apple-darwin -enable-global-merge-func=true -global-merging-skip-no-params=false < %s | FileCheck %s

; CHECK: _f1.Tgm
; CHECK: _f2.Tgm

%struct.__NSConstantString_tag = type { ptr, i32, ptr, i64 }
@__CFConstantStringClassReference = external global [0 x i32]
@.str.2 = private unnamed_addr constant [9 x i8] c"cfstring\00", section "__TEXT,__cstring,cstring_literals", align 1
@_unnamed_cfstring_ = private global %struct.__NSConstantString_tag { ptr @__CFConstantStringClassReference, i32 1992, ptr @.str.2, i64 8 }, section "__DATA,__cfstring", align 8

@.str.3 = private unnamed_addr constant [9 x i8] c"cfstring\00", section "__TEXT,__cstring,cstring_literals", align 1
@_unnamed_cfstring_.2 = private global %struct.__NSConstantString_tag { ptr @__CFConstantStringClassReference, i32 1992, ptr @.str.3, i64 8 }, section "__DATA,__cfstring", align 8

declare i32 @hoo(ptr noundef)

define i32 @f1() {
entry:
  %call = tail call i32 @hoo(ptr noundef nonnull @_unnamed_cfstring_)
  %add = sub nsw i32 %call, 1
  ret i32 %add
}

define i32 @f2() {
entry:
  %call = tail call i32 @hoo(ptr noundef nonnull @_unnamed_cfstring_.2)
  %add = sub nsw i32 %call, 1
  ret i32 %add
}
