; RUN: opt -S -mtriple=bpfel -passes=bpf-arena-cast < %s | FileCheck %s
; RUN: llc -mtriple=bpfel -mcpu=v4 -mattr=+arena < %s | FileCheck --check-prefix=ASM %s

; The type of a kfunc says what is in arena. addrspacecast in the source keeps
; the value: the program stores and passes the address as it is. The only cast
; insn is the one in front of the access.

declare ptr addrspace(1) @bpf_arena_alloc_pages(ptr, ptr addrspace(1), i32, i32, i64) section ".ksyms"
declare void @bpf_arena_free_pages(ptr, ptr addrspace(1), i32) section ".ksyms"
declare ptr @bpf_task_from_pid(i32) section ".ksyms"
@arena = external global i8, section ".maps"

; ASM-LABEL: alloc:
; ASM:       call bpf_arena_alloc_pages
; ASM-NEXT:  r1 = addr_space_cast(r0, 0, 1)
; ASM-NEXT:  *(u64 *)(r1 + 0) = r0
; ASM-NEXT:  exit
define ptr @alloc(ptr %ctx) section "syscall" {
; CHECK-LABEL: @alloc
; CHECK:       %p1 = call ptr addrspace(1) @bpf_arena_alloc_pages
; CHECK:       ptrtoint ptr addrspace(1) %p1 to i64
; CHECK:       %p = inttoptr
  %p1 = call ptr addrspace(1) @bpf_arena_alloc_pages(ptr @arena, ptr addrspace(1) null, i32 1, i32 -1, i64 0)
  %p = addrspacecast ptr addrspace(1) %p1 to ptr
  store ptr %p, ptr %p
  ret ptr %p
}

; ASM-LABEL: free:
; ASM-NOT:   addr_space_cast
; ASM:       call bpf_arena_free_pages
define void @free(ptr %p) {
  %p1 = addrspacecast ptr %p to ptr addrspace(1)
  call void @bpf_arena_free_pages(ptr @arena, ptr addrspace(1) %p1, i32 1)
  ret void
}

; All other kfuncs return pointers of the kernel.
; ASM-LABEL: task:
; ASM-NOT:   addr_space_cast
; ASM:       exit
define i32 @task(ptr %ctx) section "syscall" {
  %t = call ptr @bpf_task_from_pid(i32 1)
  %c = icmp eq ptr %t, null
  br i1 %c, label %out, label %use
use:
  %v = load i32, ptr %t
  ret i32 %v
out:
  ret i32 0
}
