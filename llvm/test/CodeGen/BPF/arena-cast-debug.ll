; REQUIRES: asserts
; RUN: opt -S -mtriple=bpfel -passes=bpf-arena-cast -debug-only=bpf-arena-cast -disable-output < %s 2>&1 | FileCheck %s

; Which pointers are cast. Helper 158 is bpf_get_current_task_btf().

@in_arena = global ptr null, section ".bss.addr_space.1"

; CHECK:      prog: cast @in_arena{{$}}
; CHECK-NEXT: prog: cast %a{{$}}
; CHECK-NEXT: prog: cast %p, can be a pointer of the kernel
; CHECK-NOT:  cast
define i64 @prog(ptr %ctx, i1 %c) section "syscall" {
  %a = load ptr, ptr @in_arena
  %x = load i64, ptr %a
  %t = call ptr inttoptr (i64 158 to ptr)()
  %y = load i64, ptr %t
  %p = select i1 %c, ptr %t, ptr %a
  %z = load i64, ptr %p
  %xy = add i64 %x, %y
  %r = add i64 %xy, %z
  ret i64 %r
}
