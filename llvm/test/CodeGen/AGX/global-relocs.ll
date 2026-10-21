; RUN: llc -mtriple=agx -mcpu=agx-g14c -filetype=obj < %s -o %t.o
; RUN: llvm-readobj -r %t.o | FileCheck %s

; Global data that holds addresses of global data gets absolute relocations
; (R_AGX_ABS64 = 1, R_AGX_ABS32 = 2), which the loader applies.

; CHECK:      Section ({{[0-9]+}}) .rela.agx.globals {
; CHECK-NEXT:   0x28 R_AGX_ABS64 x 0x0
; CHECK-NEXT:   0x30 R_AGX_ABS64 y 0x0
; CHECK-NEXT: }

@x = addrspace(1) global i64 1
@y = addrspace(1) global [4 x i64] [i64 4, i64 5, i64 6, i64 0]
@p = addrspace(1) global [2 x ptr addrspace(1)] [ptr addrspace(1) @x, ptr addrspace(1) @y]

define agx_kernel void @k(ptr addrspace(1) %out) {
  %a = load ptr addrspace(1), ptr addrspace(1) getelementptr ([2 x ptr addrspace(1)], ptr addrspace(1) @p, i32 0, i32 1)
  %v = load i64, ptr addrspace(1) %a
  store i64 %v, ptr addrspace(1) %out
  ret void
}
