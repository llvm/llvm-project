target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@object_in_asm = external global i8
@object_in_c = external constant i32

define i64 @use() {
  %a = ptrtoint ptr @object_in_asm to i64
  %b = ptrtoint ptr @object_in_c to i64
  %r = add i64 %a, %b
  ret i64 %r
}
