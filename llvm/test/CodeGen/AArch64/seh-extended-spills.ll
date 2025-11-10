; RUN: llc -mtriple aarch64-unknown-windows-msvc -filetype asm -o - %s | FileCheck %s

declare dso_local void @g(ptr noundef)
define dso_local preserve_mostcc void @f(ptr noundef %p) #0 {
entry:
  %p.addr = alloca ptr, align 8
  store ptr %p, ptr %p.addr, align 8
  %0 = load ptr, ptr %p.addr, align 8
  call void @g(ptr noundef %0)
  ret void
}

attributes #0 = { nounwind uwtable(sync) }

; CHECK: stp x9, x10, [sp, #[[OFFSET_0:[0-9]+]]]
; CHECK-NEXT: .seh_save_any_reg_p x9, [[OFFSET_0]]
; CHECK: stp x11, x12, [sp, #[[OFFSET_1:[0-9]+]]]
; CHECK-NEXT: .seh_save_any_reg_p x11, [[OFFSET_1]]
; CHECK: stp x13, x14, [sp, #[[OFFSET_2:[0-9]+]]]
; CHECK-NEXT: .seh_save_any_reg_p x13, [[OFFSET_2]]
; CHECK: str x15, [sp, #[[OFFSET_3:[0-9]+]]]
; CHECK-NEXT: .seh_save_any_reg x15, [[OFFSET_3]]
; CHECK: .seh_endprologue

; CHECK: .seh_startepilogue
; CHECK: ldr x15, [sp, #[[OFFSET_3]]]
; CHECK-NEXT: .seh_save_any_reg x15, [[OFFSET_3]]
; CHECK: ldp x13, x14, [sp, #[[OFFSET_2]]]
; CHECK-NEXT: .seh_save_any_reg_p x13, [[OFFSET_2]]
; CHECK: ldp x11, x12, [sp, #[[OFFSET_1]]]
; CHECK-NEXT: .seh_save_any_reg_p x11, [[OFFSET_1]]
; CHECK: ldp x9, x10, [sp, #[[OFFSET_0]]]
; CHECK-NEXT: .seh_save_any_reg_p x9, [[OFFSET_0]]
; CHECK: .seh_endepilogue
