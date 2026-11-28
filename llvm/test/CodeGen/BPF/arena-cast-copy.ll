; RUN: opt -S -mtriple=bpfel -passes=bpf-arena-cast < %s | FileCheck %s

; One cast insn cannot see a pointer of the kernel and a number. The caller
; that passes a map value gets a copy of the function, without a cast.

@g = internal global i64 0, section ".bss.addr_space.1"

; CHECK-LABEL: define internal i64 @read(ptr %p)
; CHECK:       addrspacecast
define internal i64 @read(ptr %p) noinline {
  %v = load i64, ptr %p
  ret i64 %v
}

; CHECK-LABEL: define i64 @prog(
; CHECK:       call i64 @read.kern(ptr %m)
; CHECK:       call i64 @read(ptr @g)
define i64 @prog(ptr %ctx) section "syscall" {
  %m = call ptr inttoptr (i64 1 to ptr)(ptr null, ptr null)
  %a = call i64 @read(ptr %m)
  %b = call i64 @read(ptr @g)
  %r = add i64 %a, %b
  ret i64 %r
}

; CHECK-LABEL: define internal i64 @read.kern(ptr %p)
; CHECK-NOT:   addrspacecast
; CHECK:       ret i64
