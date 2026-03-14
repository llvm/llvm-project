;; Check that this produces the expected assembly output
; RUN: llc -mtriple=thumbv7-windows -o - %s -verify-machineinstrs | FileCheck %s
;; Also try to write an object file, which verifies that the SEH opcodes
;; match the actual prologue/epilogue length.
; RUN: llc -mtriple=thumbv7-windows -filetype=obj -o %t.obj %s -verify-machineinstrs

; CHECK-LABEL: alloc_local:
; CHECK-NEXT: .seh_proc alloc_local
; CHECK-NEXT: @ %bb.0:                                @ %entry
; CHECK-NEXT:         push.w  {r4, r5, r6, r7, r8, r9, r10}
; CHECK-NEXT:         .seh_save_regs_w        {r4-r10}
; CHECK-NEXT:         sub     sp, #4
; CHECK-NEXT:         .seh_stackalloc 4
; CHECK-NEXT:         vpush   {d8, d9, d10, d11, d12, d13, d14, d15}
; CHECK-NEXT:         .seh_save_fregs         {d8-d15}
; CHECK-NEXT:         push.w  {r11, lr}
; CHECK-NEXT:         .seh_save_regs_w        {r11, lr}
; CHECK-NEXT:         mov     r11, sp
; CHECK-NEXT:         .seh_save_sp    r11
; CHECK-NEXT:         .seh_endprologue
; CHECK-NEXT:         movw    r4, #1256
; CHECK-NEXT:         bl      __chkstk
; CHECK-NEXT:         sub.w   sp, sp, r4
; CHECK-NEXT:         mov     r4, sp
; CHECK-NEXT:         bfc     r4, #0, #4
; CHECK-NEXT:         mov     sp, r4

; CHECK:              ldr.w   [[TMP:r[0-9]]], [r11, #104]
; CHECK:              mov     r0, [[TMP]]

; CHECK:              .seh_startepilogue
; CHECK-NEXT:         mov     sp, r11
; CHECK-NEXT:         .seh_save_sp    r11
; CHECK-NEXT:         pop.w   {r11, lr}
; CHECK-NEXT:         .seh_save_regs_w        {r11, lr}
; CHECK-NEXT:         vpop    {d8, d9, d10, d11, d12, d13, d14, d15}
; CHECK-NEXT:         .seh_save_fregs         {d8-d15}
; CHECK-NEXT:         add     sp, #4
; CHECK-NEXT:         .seh_stackalloc 4
; CHECK-NEXT:         pop.w   {r4, r5, r6, r7, r8, r9, r10}
; CHECK-NEXT:         .seh_save_regs_w        {r4-r10}
; CHECK-NEXT:         bx      lr
; CHECK-NEXT:         .seh_nop
; CHECK-NEXT:         .seh_endepilogue
; CHECK-NEXT:         .seh_endproc

define arm_aapcs_vfpcc void @alloc_local(i32 noundef %a, i32 noundef %b, i32 noundef %c, i32 noundef %d, i32 noundef %e) uwtable {
entry:
  %buf2 = alloca [5000 x i8], align 16
  %vla = alloca i8, i32 %a, align 1
  call void @llvm.lifetime.start.p0(i64 5000, ptr nonnull %buf2) #3
  call arm_aapcs_vfpcc void @other(i32 noundef %e, ptr noundef nonnull %vla, ptr noundef nonnull %buf2)
  call void asm sideeffect "", "~{r4},~{r5},~{r6},~{r7},~{r8},~{r9},~{r10},~{r11},~{r12}"()
  call void asm sideeffect "", "~{d8},~{d9},~{d10},~{d11},~{d12},~{d13},~{d14},~{d15}"()
  call void @llvm.lifetime.end.p0(i64 5000, ptr nonnull %buf2) #3
  ret void
}

declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture)

declare arm_aapcs_vfpcc void @other(i32 noundef, ptr noundef, ptr noundef)

; CHECK-LABEL: everything_varargs:
; CHECK-NEXT: .seh_proc everything_varargs
; CHECK-NEXT: @ %bb.0:                                @ %entry
; CHECK-NEXT:         sub     sp, #12
; CHECK-NEXT:         .seh_stackalloc 12
; CHECK-NEXT:         push.w  {r4, r5, r6, r7, r8, r9}
; CHECK-NEXT:         .seh_save_regs_w        {r4-r9}
; CHECK-NEXT:         sub     sp, #4
; CHECK-NEXT:         .seh_stackalloc 4
; CHECK-NEXT:         vpush   {d8, d9, d10, d11, d12, d13, d14, d15}
; CHECK-NEXT:         .seh_save_fregs         {d8-d15}
; CHECK-NEXT:         push.w  {r11, lr}
; CHECK-NEXT:         .seh_save_regs_w        {r11, lr}
; CHECK-NEXT:         mov     r11, sp
; CHECK-NEXT:         .seh_save_sp    r11
; CHECK-NEXT:         .seh_endprologue
; CHECK-NEXT:         movw    r4, #1258
; CHECK-NEXT:         bl      __chkstk
; CHECK-NEXT:         sub.w   sp, sp, r4
; CHECK-NEXT:         mov     r4, sp
; CHECK-NEXT:         bfc     r4, #0, #4
; CHECK-NEXT:         mov     sp, r4

; CHECK:              .seh_startepilogue
; CHECK-NEXT:         mov     sp, r11
; CHECK-NEXT:         .seh_save_sp    r11
; CHECK-NEXT:         pop.w   {r11, lr}
; CHECK-NEXT:         .seh_save_regs_w        {r11, lr}
; CHECK-NEXT:         vpop    {d8, d9, d10, d11, d12, d13, d14, d15}
; CHECK-NEXT:         .seh_save_fregs         {d8-d15}
; CHECK-NEXT:         add     sp, #4
; CHECK-NEXT:         .seh_stackalloc 4
; CHECK-NEXT:         pop.w   {r4, r5, r6, r7, r8, r9}
; CHECK-NEXT:         .seh_save_regs_w        {r4-r9}
; CHECK-NEXT:         add     sp, #12
; CHECK-NEXT:         .seh_stackalloc 12
; CHECK-NEXT:         bx      lr
; CHECK-NEXT:         .seh_nop
; CHECK-NEXT:         .seh_endepilogue
; CHECK-NEXT:         .seh_endproc

define arm_aapcs_vfpcc void @everything_varargs(i32 noundef %a, ...) {
entry:
  %buf2 = alloca [5000 x i8], align 16
  %ap = alloca ptr, align 4
  %vla = alloca i8, i32 %a, align 1
  call void @llvm.lifetime.start.p0(i64 5000, ptr nonnull %buf2)
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %ap)
  call void @llvm.va_start(ptr nonnull %ap)
  %0 = load ptr, ptr %ap, align 4
  call arm_aapcs_vfpcc void @other2(i32 noundef %a, ptr noundef nonnull %vla, ptr noundef nonnull %buf2, ptr noundef %0)
  call void @llvm.va_end(ptr nonnull %ap)
  call void asm sideeffect "", "~{r4},~{r5},~{r6},~{r7},~{r8},~{r9},~{r11},~{r12}"()
  call void asm sideeffect "", "~{d8},~{d9},~{d10},~{d11},~{d12},~{d13},~{d14},~{d15}"()
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %ap)
  call void @llvm.lifetime.end.p0(i64 5000, ptr nonnull %buf2)
  ret void
}

; CHECK-LABEL: novector_varargs:
; CHECK-NEXT: .seh_proc novector_varargs
; CHECK-NEXT: @ %bb.0:                                @ %entry
; CHECK-NEXT:         sub     sp, #12
; CHECK-NEXT:         .seh_stackalloc 12
; CHECK-NEXT:         push.w  {r4, r5, r6, r7, r8, r9}
; CHECK-NEXT:         .seh_save_regs_w        {r4-r9}
; CHECK-NEXT:         push.w  {r11, lr}
; CHECK-NEXT:         .seh_save_regs_w        {r11, lr}
; CHECK-NEXT:         mov     r11, sp
; CHECK-NEXT:         .seh_save_sp    r11
; CHECK-NEXT:         .seh_endprologue
; CHECK-NEXT:         movw    r4, #1259
; CHECK-NEXT:         bl      __chkstk
; CHECK-NEXT:         sub.w   sp, sp, r4
; CHECK-NEXT:         mov     r4, sp
; CHECK-NEXT:         bfc     r4, #0, #4
; CHECK-NEXT:         mov     sp, r4

; CHECK:              .seh_startepilogue
; CHECK-NEXT:         mov     sp, r11
; CHECK-NEXT:         .seh_save_sp    r11
; CHECK-NEXT:         pop.w   {r11, lr}
; CHECK-NEXT:         .seh_save_regs_w        {r11, lr}
; CHECK-NEXT:         pop.w   {r4, r5, r6, r7, r8, r9}
; CHECK-NEXT:         .seh_save_regs_w        {r4-r9}
; CHECK-NEXT:         add     sp, #12
; CHECK-NEXT:         .seh_stackalloc 12
; CHECK-NEXT:         bx      lr
; CHECK-NEXT:         .seh_nop
; CHECK-NEXT:         .seh_endepilogue
; CHECK-NEXT:         .seh_endproc

define arm_aapcs_vfpcc void @novector_varargs(i32 noundef %a, ...) {
entry:
  %buf2 = alloca [5000 x i8], align 16
  %ap = alloca ptr, align 4
  %vla = alloca i8, i32 %a, align 1
  call void @llvm.lifetime.start.p0(i64 5000, ptr nonnull %buf2)
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %ap)
  call void @llvm.va_start(ptr nonnull %ap)
  %0 = load ptr, ptr %ap, align 4
  call arm_aapcs_vfpcc void @other2(i32 noundef %a, ptr noundef nonnull %vla, ptr noundef nonnull %buf2, ptr noundef %0)
  call void @llvm.va_end(ptr nonnull %ap)
  call void asm sideeffect "", "~{r4},~{r5},~{r6},~{r7},~{r8},~{r9},~{r11},~{r12}"()
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %ap)
  call void @llvm.lifetime.end.p0(i64 5000, ptr nonnull %buf2)
  ret void
}

declare void @llvm.va_start(ptr)
declare void @llvm.va_end(ptr)

declare arm_aapcs_vfpcc void @other2(i32 noundef, ptr noundef, ptr noundef, ptr noundef)
