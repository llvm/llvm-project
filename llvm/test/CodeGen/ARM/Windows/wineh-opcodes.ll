;; Check that this produces the expected assembly output
; RUN: llc -mtriple=thumbv7-windows -o - %s -verify-machineinstrs | FileCheck %s
;; Also try to write an object file, which verifies that the SEH opcodes
;; match the actual prologue/epilogue length.
; RUN: llc -mtriple=thumbv7-windows -filetype=obj -o %t.obj %s -verify-machineinstrs

; CHECK-LABEL: clobberR4Frame:
; CHECK-NEXT: .seh_proc clobberR4Frame
; CHECK-NEXT: @ %bb.0:                                @ %entry
; CHECK-NEXT:         push.w  {r4, r7, r11, lr}
; CHECK-NEXT:         .seh_save_regs_w        {r4, r7, r11, lr}
; CHECK-NEXT:         add.w   r11, sp, #8
; CHECK-NEXT:         .seh_nop_w
; CHECK-NEXT:         .seh_endprologue
; CHECK-NEXT:         bl      other

; CHECK:              .seh_startepilogue
; CHECK-NEXT:         pop.w   {r4, r7, r11, pc}
; CHECK-NEXT:         .seh_save_regs_w        {r4, r7, r11, lr}
; CHECK-NEXT:         .seh_endepilogue
; CHECK-NEXT:         .seh_endproc

define arm_aapcs_vfpcc void @clobberR4Frame() uwtable "frame-pointer"="all" {
entry:
  call arm_aapcs_vfpcc void @other()
  call void asm sideeffect "", "~{r4}"()
  ret void
}

; CHECK-LABEL: clobberR4NoFrame:
; CHECK-NEXT: .seh_proc clobberR4NoFrame
; CHECK-NEXT: @ %bb.0:                                @ %entry
; CHECK-NEXT:         push    {r4, lr}
; CHECK-NEXT:         .seh_save_regs  {r4, lr}
; CHECK-NEXT:         .seh_endprologue
; CHECK-NEXT:         bl      other

; CHECK:              .seh_startepilogue
; CHECK-NEXT:         pop     {r4, pc}
; CHECK-NEXT:         .seh_save_regs  {r4, lr}
; CHECK-NEXT:         .seh_endepilogue
; CHECK-NEXT:         .seh_endproc

define arm_aapcs_vfpcc void @clobberR4NoFrame() uwtable "frame-pointer"="none" {
entry:
  call arm_aapcs_vfpcc void @other()
  call void asm sideeffect "", "~{r4}"()
  ret void
}

; CHECK-LABEL: clobberR4Tail:
; CHECK-NEXT: .seh_proc clobberR4Tail
; CHECK-NEXT: @ %bb.0:                                @ %entry
; CHECK-NEXT:         push    {r4, lr}
; CHECK-NEXT:         .seh_save_regs  {r4, lr}
; CHECK-NEXT:         .seh_endprologue

; CHECK:              .seh_startepilogue
; CHECK-NEXT:         pop.w   {r4, lr}
; CHECK-NEXT:         .seh_save_regs_w        {r4, lr}
; CHECK-NEXT:         b.w     other
; CHECK-NEXT:         .seh_nop_w
; CHECK-NEXT:         .seh_endepilogue
; CHECK-NEXT:         .seh_endproc

define arm_aapcs_vfpcc void @clobberR4Tail() uwtable "frame-pointer"="none" {
entry:
  call void asm sideeffect "", "~{r4}"()
  tail call arm_aapcs_vfpcc void @other()
  ret void
}

; CHECK-LABEL: clobberD8D10:
; CHECK-NEXT: .seh_proc clobberD8D10
; CHECK-NEXT: @ %bb.0:                                @ %entry
; CHECK-NEXT:         vpush   {d8, d9, d10}
; CHECK-NEXT:         .seh_save_fregs {d8-d10}
; CHECK-NEXT:         .seh_endprologue

; CHECK:              .seh_startepilogue
; CHECK-NEXT:         vpop    {d8, d9, d10}
; CHECK-NEXT:         .seh_save_fregs {d8-d10}
; CHECK-NEXT:         b.w     other
; CHECK-NEXT:         .seh_nop_w
; CHECK-NEXT:         .seh_endepilogue
; CHECK-NEXT:         .seh_endproc

define arm_aapcs_vfpcc void @clobberD8D10() uwtable "frame-pointer"="none" {
entry:
  call void asm sideeffect "", "~{d8},~{d9},~{d10}"()
  tail call arm_aapcs_vfpcc void @other()
  ret void
}

declare arm_aapcs_vfpcc void @other()

; CHECK-LABEL: vararg:
; CHECK-NEXT: .seh_proc vararg
; CHECK-NEXT: @ %bb.0:                                @ %entry
; CHECK-NEXT:         sub     sp, #12
; CHECK-NEXT:         .seh_stackalloc 12
; CHECK-NEXT:         push.w  {r11, lr}
; CHECK-NEXT:         .seh_save_regs_w        {r11, lr}
; CHECK-NEXT:         sub     sp, #4
; CHECK-NEXT:         .seh_stackalloc 4
; CHECK-NEXT:         .seh_endprologue

; CHECK:              .seh_startepilogue
; CHECK-NEXT:         add     sp, #4
; CHECK-NEXT:         .seh_stackalloc 4
; CHECK-NEXT:         pop.w   {r11, lr}
; CHECK-NEXT:         .seh_save_regs_w        {r11, lr}
; CHECK-NEXT:         add     sp, #12
; CHECK-NEXT:         .seh_stackalloc 12
; CHECK-NEXT:         bx      lr
; CHECK-NEXT:         .seh_nop
; CHECK-NEXT:         .seh_endepilogue
; CHECK-NEXT:         .seh_endproc

define arm_aapcs_vfpcc void @vararg(i32 noundef %a, ...) uwtable "frame-pointer"="none" {
entry:
  %ap = alloca ptr, align 4
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %ap)
  call void @llvm.va_start(ptr nonnull %ap)
  %0 = load ptr, ptr %ap
  call arm_aapcs_vfpcc void @useva(ptr noundef %0)
  call void @llvm.va_end(ptr nonnull %ap)
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %ap)
  ret void
}

declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture)
declare void @llvm.va_start(ptr)
declare void @llvm.va_end(ptr)

declare arm_aapcs_vfpcc void @useva(ptr noundef)

; CHECK-LABEL: onlystack:
; CHECK-NEXT: .seh_proc onlystack
; CHECK-NEXT: @ %bb.0:                                @ %entry
; CHECK-NEXT:         sub     sp, #4
; CHECK-NEXT:         .seh_stackalloc 4
; CHECK-NEXT:         .seh_endprologue

; CHECK:              .seh_startepilogue
; CHECK-NEXT:         add     sp, #4
; CHECK-NEXT:         .seh_stackalloc 4
; CHECK-NEXT:         bx      lr
; CHECK-NEXT:         .seh_nop
; CHECK-NEXT:         .seh_endepilogue
; CHECK-NEXT:         .seh_endproc

define dso_local arm_aapcs_vfpcc void @onlystack() uwtable "frame-pointer"="none" {
entry:
  %buf = alloca [4 x i8], align 1
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %buf)
  call void asm sideeffect "", "r"(ptr nonnull %buf)
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %buf)
  ret void
}

; CHECK-LABEL: func50:
; CHECK-NEXT: .seh_proc func50
; CHECK-NEXT: @ %bb.0:                                @ %entry
; CHECK-NEXT:         push.w  {r11, lr}
; CHECK-NEXT:         .seh_save_regs_w        {r11, lr}
; CHECK-NEXT:         sub     sp, #56
; CHECK-NEXT:         .seh_stackalloc 56
; CHECK-NEXT:         .seh_endprologue

; CHECK:              .seh_startepilogue
; CHECK-NEXT:         add     sp, #56
; CHECK-NEXT:         .seh_stackalloc 56
; CHECK-NEXT:         pop.w   {r11, pc}
; CHECK-NEXT:         .seh_save_regs_w        {r11, lr}
; CHECK-NEXT:         .seh_endepilogue
; CHECK-NEXT:         .seh_endproc

define arm_aapcs_vfpcc void @func50() {
entry:
  %buf = alloca [50 x i8], align 1
  call void @llvm.lifetime.start.p0(i64 50, ptr nonnull %buf)
  call arm_aapcs_vfpcc void @useptr(ptr noundef nonnull %buf)
  call void @llvm.lifetime.end.p0(i64 50, ptr nonnull %buf)
  ret void
}

; CHECK-LABEL: func4000:
; CHECK-NEXT: .seh_proc func4000
; CHECK-NEXT: @ %bb.0:                                @ %entry
; CHECK-NEXT:         push.w  {r11, lr}
; CHECK-NEXT:         .seh_save_regs_w        {r11, lr}
; CHECK-NEXT:         sub.w   sp, sp, #4000
; CHECK-NEXT:         .seh_stackalloc_w       4000
; CHECK-NEXT:         .seh_endprologue

; CHECK:              .seh_startepilogue
; CHECK-NEXT:         add.w   sp, sp, #4000
; CHECK-NEXT:         .seh_stackalloc_w       4000
; CHECK-NEXT:         pop.w   {r11, pc}
; CHECK-NEXT:         .seh_save_regs_w        {r11, lr}
; CHECK-NEXT:         .seh_endepilogue
; CHECK-NEXT:         .seh_endproc

define arm_aapcs_vfpcc void @func4000() {
entry:
  %buf = alloca [4000 x i8], align 1
  call void @llvm.lifetime.start.p0(i64 4000, ptr nonnull %buf)
  call arm_aapcs_vfpcc void @useptr(ptr noundef nonnull %buf)
  call void @llvm.lifetime.end.p0(i64 4000, ptr nonnull %buf)
  ret void
}

; CHECK-LABEL: func5000:
; CHECK-NEXT: .seh_proc func5000
; CHECK-NEXT: @ %bb.0:                                @ %entry
; CHECK-NEXT:         push    {r4, r5, r6, lr}
; CHECK-NEXT:         .seh_save_regs  {r4-r6, lr}
; CHECK-NEXT:         movw    r4, #1250
; CHECK-NEXT:         .seh_nop_w
; CHECK-NEXT:         bl      __chkstk
; CHECK-NEXT:         .seh_nop_w
; CHECK-NEXT:         sub.w   sp, sp, r4
; CHECK-NEXT:         .seh_stackalloc_w       5000
; CHECK-NEXT:         .seh_endprologue

; CHECK:              .seh_startepilogue
; CHECK-NEXT:         add.w   sp, sp, #4992
; CHECK-NEXT:         .seh_stackalloc_w       4992
; CHECK-NEXT:         add     sp, #8
; CHECK-NEXT:         .seh_stackalloc 8
; CHECK-NEXT:         pop     {r4, r5, r6, pc}
; CHECK-NEXT:         .seh_save_regs  {r4-r6, lr}
; CHECK-NEXT:         .seh_endepilogue
; CHECK-NEXT:         .seh_endproc

define arm_aapcs_vfpcc void @func5000() {
entry:
  %buf = alloca [5000 x i8], align 1
  call void @llvm.lifetime.start.p0(i64 5000, ptr nonnull %buf)
  call arm_aapcs_vfpcc void @useptr(ptr noundef nonnull %buf)
  call void @llvm.lifetime.end.p0(i64 5000, ptr nonnull %buf)
  ret void
}

; CHECK-LABEL: func262144:
; CHECK-NEXT: .seh_proc func262144
; CHECK-NEXT: @ %bb.0:                                @ %entry
; CHECK-NEXT:         push    {r4, r5, r6, lr}
; CHECK-NEXT:         .seh_save_regs  {r4-r6, lr}
; CHECK-NEXT:         movs    r4, #0
; CHECK-NEXT:         .seh_nop
; CHECK-NEXT:         movt    r4, #1
; CHECK-NEXT:         .seh_nop_w
; CHECK-NEXT:         bl      __chkstk
; CHECK-NEXT:         .seh_nop_w
; CHECK-NEXT:         sub.w   sp, sp, r4
; CHECK-NEXT:         .seh_stackalloc_w       262144
; CHECK-NEXT:         .seh_endprologue

; CHECK:              .seh_startepilogue
; CHECK-NEXT:         add.w   sp, sp, #262144
; CHECK-NEXT:         .seh_stackalloc_w       262144
; CHECK-NEXT:         pop     {r4, r5, r6, pc}
; CHECK-NEXT:         .seh_save_regs  {r4-r6, lr}
; CHECK-NEXT:         .seh_endepilogue
; CHECK-NEXT:         .seh_endproc

define arm_aapcs_vfpcc void @func262144() {
entry:
  %buf = alloca [262144 x i8], align 1
  call void @llvm.lifetime.start.p0(i64 262144, ptr nonnull %buf)
  call arm_aapcs_vfpcc void @useptr(ptr noundef nonnull %buf)
  call void @llvm.lifetime.end.p0(i64 262144, ptr nonnull %buf)
  ret void
}

; CHECK-LABEL: func270000:
; CHECK-NEXT: .seh_proc func270000
; CHECK-NEXT: @ %bb.0:                                @ %entry
; CHECK-NEXT:         push    {r4, r5, r6, lr}
; CHECK-NEXT:         .seh_save_regs  {r4-r6, lr}
; CHECK-NEXT:         movw    r4, #1964
; CHECK-NEXT:         .seh_nop_w
; CHECK-NEXT:         movt    r4, #1
; CHECK-NEXT:         .seh_nop_w
; CHECK-NEXT:         bl      __chkstk
; CHECK-NEXT:         .seh_nop_w
; CHECK-NEXT:         sub.w   sp, sp, r4
; CHECK-NEXT:         .seh_stackalloc_w       270000
; CHECK-NEXT:         .seh_endprologue

; CHECK:              .seh_startepilogue
; CHECK-NEXT:         add.w   sp, sp, #268288
; CHECK-NEXT:         .seh_stackalloc_w       268288
; CHECK-NEXT:         add.w   sp, sp, #1712
; CHECK-NEXT:         .seh_stackalloc_w       1712
; CHECK-NEXT:         pop     {r4, r5, r6, pc}
; CHECK-NEXT:         .seh_save_regs  {r4-r6, lr}
; CHECK-NEXT:         .seh_endepilogue
; CHECK-NEXT:         .seh_endproc

define arm_aapcs_vfpcc void @func270000() {
entry:
  %buf = alloca [270000 x i8], align 1
  call void @llvm.lifetime.start.p0(i64 270000, ptr nonnull %buf)
  call arm_aapcs_vfpcc void @useptr(ptr noundef nonnull %buf)
  call void @llvm.lifetime.end.p0(i64 270000, ptr nonnull %buf)
  ret void
}

declare arm_aapcs_vfpcc void @useptr(ptr noundef)
