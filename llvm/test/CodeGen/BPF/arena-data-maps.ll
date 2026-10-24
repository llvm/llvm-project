; RUN: opt -mtriple=bpfel -passes=bpf-arena-data < %s \
; RUN:   | llc -mtriple=bpfel -mcpu=v4 -mattr=+arena-data -filetype=obj -o %t
; RUN: llvm-readelf -S %t | FileCheck %s
;
; The map that is added for the data is in the section of the map that the
; program defines. libbpf looks at one section with that name.

; CHECK-COUNT-1: .maps
; CHECK-NOT:     .maps

@m = global { ptr, ptr } zeroinitializer, section ".maps", align 8
@g = global i32 1
@llvm.compiler.used = appending global [1 x ptr] [ptr @m], section "llvm.metadata"

define i32 @f(ptr %c) section "syscall" {
  %v = load i32, ptr @g
  ret i32 %v
}
