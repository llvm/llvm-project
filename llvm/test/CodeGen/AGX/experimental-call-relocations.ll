; RUN: llc -mtriple=agx -mcpu=agx-g14c -agx-experimental-leaf-calls -agx-call-relocations -filetype=obj < %s \
; RUN:   | llvm-readelf -r -s - | FileCheck %s
; With -agx-call-relocations a call to a function is a relocation (type 3,
; R_AGX_CALL32: S + A - P + 2 into the call's 32-bit field, two bytes into the
; instruction), for a linker that places the functions (ZLUDA's object merger);
; branches within a function stay resolved.

; CHECK: Relocation section '.rela.text'
; CHECK-NEXT: Offset
; CHECK-NEXT: 0000000000000016  {{[0-9a-f]+}}00000003 {{.*}} .text + 3a
; CHECK: 000000000000003a {{.*}} FUNC LOCAL {{.*}} callee

define internal i32 @callee(i32 %x) noinline {
  %y = mul i32 %x, %x
  ret i32 %y
}

define agx_kernel void @kernel(ptr addrspace(1) %out, i32 %x) {
  %y = call i32 @callee(i32 %x)
  store i32 %y, ptr addrspace(1) %out, align 4
  ret void
}
