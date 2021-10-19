; RUN: llc < %s -mtriple=armv7-none-linux- | FileCheck %s
; Check that LDRB_POST_IMM instruction emitted properly.

%my_struct_t = type { i8, i8, i8, i8, i8 }
@main.val = private unnamed_addr constant %my_struct_t { i8 1, i8 2, i8 3, i8 4, i8 5 }

declare void @f(i32 %n1, i32 %n2, i32 %n3, %my_struct_t* byval(%my_struct_t) %val);

; CHECK-LABEL: main:
define i32 @main() nounwind {
entry:
; CHECK: ldrb	{{(r[0-9]+)}}, {{(\[r[0-9]+\])}}, #1
  call void @f(i32 555, i32 555, i32 555, %my_struct_t* byval(%my_struct_t) @main.val)
  ret i32 0
}

