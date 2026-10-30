; An address made by inttoptr and used with two pointees (a memset of its
; bytes, an i32 store) is one inttoptr per use: the AIR writer gives each
; pointer value one pointee type, and Metal's reader rejects a pointer used
; as two ("Unexpected bitcode file", Cycles' intersect_volume_stack).
; RUN: llc -mtriple=air -filetype=obj -air-output=air %s -o %t.air
; RUN: llvm-dis %t.air -o - | FileCheck %s

; CHECK-LABEL: define void @k(
; CHECK: inttoptr
; CHECK: inttoptr
; CHECK: ret void

define void @k(ptr addrspace(1) %out, i64 %a, i32 %v) "air-kernel" {
  %p = inttoptr i64 %a to ptr addrspace(1)
  call void @llvm.memset.p1.i64(ptr addrspace(1) %p, i8 -1, i64 16, i1 false)
  store i32 %v, ptr addrspace(1) %p, align 4
  ret void
}

declare void @llvm.memset.p1.i64(ptr addrspace(1), i8, i64, i1)
