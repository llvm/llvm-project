; RUN: llc -mtriple=aarch64 -aarch64-min-jump-table-entries=4 -stack-size-section %s -o - | FileCheck %s

; CHECK-LABEL: .section .stack_sizes,"o",@progbits,.text{{$}}
; CHECK-NEXT: .xword .Lfunc_begin0
; CHECK-NEXT: .byte 0
define void @empty() {
  ret void
}

; CHECK-LABEL: .section .stack_sizes,"o",@progbits,.text{{$}}
; CHECK-NEXT: .xword .Lfunc_begin1
; CHECK-NEXT: .ascii "\200\001"
define void @non_empty() #0 {
  alloca [32 x i32]
  ret void
}

; CHECK-LABEL: dynalloc:
; CHECK-NOT: .section .stack_sizes
define void @dynalloc(i32 %n) #0 {
  alloca i32, i32 %n
  ret void
}

; Check that .stack_sizes section is linked to the function's section (.text),
; and not to the section containing the jump table (.rodata).
; CHECK-LABEL: linked_section:
; CHECK: .section .rodata,"a",@progbits
; CHECK: .section .stack_sizes,"o",@progbits,.text
; CHECK-NEXT: .xword .Lfunc_begin3
; CHECK-NEXT: .ascii "\220\001"
declare void @case0()
declare void @case1()
declare void @case2()
declare void @case3()
define void @linked_section(i32 %x) {
  %arr = alloca [32 x i32]
  switch i32 %x, label %sw.epilog [
    i32 0, label %sw.bb0
    i32 1, label %sw.bb1
    i32 2, label %sw.bb2
    i32 3, label %sw.bb3
  ]

sw.bb0:
  call void @case0()
  ret void

sw.bb1:
  call void @case1()
  ret void

sw.bb2:
  call void @case2()
  ret void

sw.bb3:
  call void @case3()
  ret void

sw.epilog:
  ret void
}
