; A generic address of a threadgroup variable used as an integer (PTX's
; cvta.shared then a ptrtoint, as constant expressions) crashed Metal's
; compiler: the generic cast, buried in an integer constant expression, was
; left for Metal. The Metal pipeline makes such constant expressions
; instructions; the generic address is the threadgroup pointer's integer with
; bit 63 set (generic-pointers.ll).
; RUN: opt -mtriple=air -passes=air-legalize-unsupported-ir -S %s | FileCheck %s
; RUN: llc -mtriple=air -filetype=obj %s -o /dev/null

@s = internal addrspace(3) global i32 undef, align 4

; CHECK-LABEL: define void @k(
; CHECK: [[G:%.*]] = or i64 ptrtoint (ptr addrspace(3) @s to i64), -9223372036854775808
; CHECK: store i64 {{%.*}}, ptr addrspace(1) %o
; CHECK-NOT: addrspace(4)
define void @k(ptr addrspace(1) %o) "air-kernel" {
  store i64 ptrtoint (ptr addrspace(4) addrspacecast (ptr addrspace(3) @s to ptr addrspace(4)) to i64), ptr addrspace(1) %o
  ret void
}
