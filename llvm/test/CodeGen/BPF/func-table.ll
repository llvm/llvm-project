; RUN: opt -S -mtriple=bpfel -passes=bpf-func-table %s | FileCheck %s
; RUN: llc -mtriple=bpfel -mcpu=v4 -mattr=+func-table %s -o - | FileCheck --check-prefix=ASM %s
; RUN: llc -mtriple=bpfel -mcpu=v4 %s -o - | FileCheck --check-prefix=OFF %s

; The address of a function is its index in the table.
; CHECK: @vtable = internal constant { i64, ptr, ptr } { i64 8, ptr inttoptr (i64 1 to ptr), ptr inttoptr (i64 4 to ptr) }
@vtable = internal constant { i64, ptr, ptr } { i64 8, ptr @a1, ptr @b1 }
; The address of a program is not.
; CHECK: @ops = constant { ptr } { ptr @prog }, section ".struct_ops.link"
@ops = constant { ptr } { ptr @prog }, section ".struct_ops.link"
; CHECK: @llvm.used = appending global [1 x ptr] [ptr @a2]
@llvm.used = appending global [1 x ptr] [ptr @a2], section "llvm.metadata"
; Functions of one type are next to each other.
; CHECK: @__bpf_func_table = internal constant [5 x ptr] [ptr null, ptr @a1, ptr @a2, ptr @a3, ptr @b1]

define internal i32 @a1(ptr %p) {
  ret i32 1
}
define internal i32 @a2(ptr %p) {
  ret i32 2
}
define internal void @b1(i64 %x) {
  ret void
}
define internal i32 @a3(ptr %p) {
  ret i32 3
}
define internal void @not_taken(i64 %x) {
  ret void
}

define i32 @prog(ptr %ctx) section "struct_ops/prog" {
; CHECK-LABEL: define i32 @prog(
; CHECK:         %p = load ptr, ptr %ctx
; CHECK-NEXT:    %[[I:.*]] = ptrtoint ptr %p to i64
; CHECK-NEXT:    %[[REL:.*]] = sub i64 %[[I]], 1
; CHECK-NEXT:    %[[MIN:.*]] = call i64 @llvm.umin.i64(i64 %[[REL]], i64 2)
; CHECK-NEXT:    %[[SLOT:.*]] = getelementptr inbounds ptr, ptr getelementptr inbounds (i8, ptr @__bpf_func_table, i64 8), i64 %[[MIN]]
; CHECK-NEXT:    %[[FN:.*]] = load ptr, ptr %[[SLOT]]
; CHECK-NEXT:    %r = call i32 %[[FN]](ptr %ctx)
  %p = load ptr, ptr %ctx
  %r = call i32 %p(ptr %ctx)
  %q = getelementptr i8, ptr %ctx, i64 8
  %f = load ptr, ptr %q
; One function of the type.
; CHECK:         call void @b1(i64 1)
  call void %f(i64 1)
; CHECK:         call void @not_taken(i64 2)
  call void @not_taken(i64 2)
; CHECK:         store ptr inttoptr (i64 2 to ptr), ptr %ctx
; CHECK:         store ptr inttoptr (i64 3 to ptr), ptr %q
  store ptr @a2, ptr %ctx
  store ptr @a3, ptr %q
; A helper.
; CHECK:         call i64 inttoptr (i64 5 to ptr)()
  %h = call i64 inttoptr (i64 5 to ptr)()
; No function of the type.
; CHECK:         call i64 null(i64 1, i64 2)
  %n = load ptr, ptr %q
  %z = call i64 %n(i64 1, i64 2)
  ret i32 %r
}

; ASM-LABEL: prog:
; ASM:         r1 = *(u64 *)(r6 + 0)
; ASM-NEXT:    r1 += -1
; ASM-NEXT:    if r1 < 2 goto
; ASM:         r1 = 2
; ASM:         r1 <<= 3
; ASM-NEXT:    r2 = __bpf_func_table ll
; ASM-NEXT:    r2 += r1
; ASM-NEXT:    r2 = *(u64 *)(r2 + 8)
; ASM:         callx r2
; ASM:         call b1
; There are no pointers in it anymore.
; ASM:         .section .rodata
; ASM:       vtable:
; ASM:         .section .data.rel.ro
; ASM:       __bpf_func_table:
; ASM-NEXT:    .quad 0
; ASM-NEXT:    .quad a1

; OFF-NOT: __bpf_func_table
