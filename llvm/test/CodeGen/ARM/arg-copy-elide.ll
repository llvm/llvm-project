; RUN: llc -mtriple=armv7-linux < %s | FileCheck %s

declare arm_aapcscc void @addrof_i32(ptr)
declare arm_aapcscc void @addrof_i64(ptr)

define arm_aapcscc void @simple(i32, i32, i32, i32, i32 %x) {
entry:
  %x.addr = alloca i32
  store i32 %x, ptr %x.addr
  call void @addrof_i32(ptr %x.addr)
  ret void
}

; CHECK-LABEL: simple:
; CHECK: push {r11, lr}
; CHECK: add r0, sp, #8
; CHECK: bl addrof_i32
; CHECK: pop {r11, pc}


; We need to load %x before calling addrof_i32 now because it could mutate %x in
; place.

define arm_aapcscc i32 @use_arg(i32, i32, i32, i32, i32 %x) {
entry:
  %x.addr = alloca i32
  store i32 %x, ptr %x.addr
  call void @addrof_i32(ptr %x.addr)
  ret i32 %x
}

; CHECK-LABEL: use_arg:
; CHECK: push {[[csr:[^ ]*]], lr}
; CHECK: add r0, sp, #8
; CHECK: ldr [[csr]], [sp, #8]
; CHECK: bl addrof_i32
; CHECK: mov r0, [[csr]]
; CHECK: pop {[[csr]], pc}


define arm_aapcscc i64 @split_i64(i32, i32, i32, i32, i64 %x) {
entry:
  %x.addr = alloca i64, align 4
  store i64 %x, ptr %x.addr, align 4
  call void @addrof_i64(ptr %x.addr)
  ret i64 %x
}

; CHECK-LABEL: split_i64:
; CHECK: push    {r4, r5, r11, lr}
; CHECK: sub     sp, sp, #8
; CHECK: ldr     r4, [sp, #28]
; CHECK: mov     r0, sp
; CHECK: ldr     r5, [sp, #24]
; CHECK: str     r4, [sp, #4]
; CHECK: str     r5, [sp]
; CHECK: bl      addrof_i64
; CHECK: mov     r0, r5
; CHECK: mov     r1, r4
; CHECK: add     sp, sp, #8
; CHECK: pop     {r4, r5, r11, pc}
