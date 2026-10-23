; RUN: llc -mtriple=agx -mcpu=agx-g14c -agx-experimental-leaf-calls -verify-machineinstrs < %s | FileCheck %s --check-prefix=CALL
; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s --check-prefix=INLINE
;
; Opt-in codegen keeps a safe leaf separate even without an explicit noinline
; attribute. The default backend still inlines it into the kernel.
; CALL-LABEL: kernel:
; CALL: call leaf
; CALL-NEXT: pop_exec r0l, 2
; CALL-LABEL: leaf:
; CALL: ret r1
; INLINE-LABEL: kernel:
; INLINE-NOT: call leaf
; INLINE-NOT: leaf:

define agx_kernel void @kernel() {
entry:
  %x = call i32 @leaf(i32 37)
  store volatile i32 %x, ptr addrspace(1) inttoptr (i64 4096 to ptr addrspace(1))
  ret void
}

define internal i32 @leaf(i32 %x) {
entry:
  %y = add i32 %x, 5
  ret i32 %y
}
