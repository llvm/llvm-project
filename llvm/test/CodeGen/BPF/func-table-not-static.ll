; RUN: not llc -mtriple=bpfel -mcpu=v4 -mattr=+func-table %s -o /dev/null 2>&1 | FileCheck %s
;
; callx takes static functions only, so @glob cannot be in the table. Left as
; an address in @vtable it would be an index out of range: a call of @s2.

; CHECK:     error: {{.*}}in function glob i32 (ptr): BPF func-table cannot take the address of a function that is not static
; CHECK-NOT: error:

@vtable = internal constant [3 x ptr] [ptr @glob, ptr @s1, ptr @s2]

define i32 @glob(ptr %p) {
  ret i32 0
}
define internal i32 @s1(ptr %p) {
  ret i32 1
}
define internal i32 @s2(ptr %p) {
  ret i32 2
}

; A call of such function is fine.
define i32 @called(ptr %p) {
  ret i32 3
}

define i32 @prog(ptr %ctx, i64 %i) section "socket" {
  %slot = getelementptr ptr, ptr @vtable, i64 %i
  %f = load ptr, ptr %slot
  %r = call i32 %f(ptr %ctx)
  %d = call i32 @called(ptr %ctx)
  %s = add i32 %r, %d
  ret i32 %s
}
