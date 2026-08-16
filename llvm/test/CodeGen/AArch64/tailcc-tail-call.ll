; RUN: llc -verify-machineinstrs < %s -mtriple=aarch64-none-linux-gnu | FileCheck %s --check-prefixes=SDAG,COMMON
; RUN: llc -global-isel -global-isel-abort=1 -verify-machineinstrs < %s -mtriple=aarch64-none-linux-gnu  | FileCheck %s --check-prefixes=GISEL,COMMON

declare tailcc void @callee_stack0()
declare tailcc void @callee_stack8([8 x i64], i64)
declare tailcc void @callee_stack16([8 x i64], i64, i64)
declare extern_weak tailcc void @callee_weak()

define tailcc void @caller_to0_from0() nounwind {
; COMMON-LABEL: caller_to0_from0:
; COMMON-NEXT: // %bb.

  tail call tailcc void @callee_stack0()
  ret void

; COMMON-NEXT: b callee_stack0
}

define tailcc void @caller_to0_from8([8 x i64], i64) #0 {
; COMMON-LABEL: caller_to0_from8:

  tail call tailcc void @callee_stack0()
  ret void

; COMMON: add sp, sp, #16
; COMMON: .cfi_def_cfa_offset -16
; COMMON-NEXT: b callee_stack0
}

define tailcc void @caller_to8_from0() "frame-pointer"="all" uwtable {
; COMMON-LABEL: caller_to8_from0:

; Key point is that the "42" should go #16 below incoming stack
; pointer (we didn't have arg space to reuse).
  tail call tailcc void @callee_stack8([8 x i64] undef, i64 42)
  ret void

; COMMON: str {{x[0-9]+}}, [x29, #16]
; COMMON: ldp x29, x30, [sp], #16
  ; If there is a sub here then the 42 will be briefly exposed to corruption
  ; from an interrupt if the kernel does not honour a red-zone, and a larger
  ; call could well overflow the red zone even if it is present.
; COMMON-NOT: sub sp,
; COMMON-NEXT: .cfi_def_cfa_offset 16
; COMMON-NEXT: .cfi_restore w30
; COMMON-NEXT: .cfi_restore w29
; COMMON-NEXT: b callee_stack8
}

define tailcc void @caller_to8_from8([8 x i64], i64 %a) #0 {
; COMMON-LABEL: caller_to8_from8:
; COMMON-NOT: sub sp,

; Key point is that the "%a" should go where at SP on entry.
  tail call tailcc void @callee_stack8([8 x i64] undef, i64 42)
  ret void

; COMMON: str {{x[0-9]+}}, [sp]
; COMMON-NEXT: b callee_stack8
}

define tailcc void @caller_to16_from8([8 x i64], i64 %a) #0 {
; COMMON-LABEL: caller_to16_from8:
; COMMON-NOT: sub sp,

; Important point is that the call reuses the "dead" argument space
; above %a on the stack. If it tries to go below incoming-SP then the
; callee will not deallocate the space, even in tailcc.
  tail call tailcc void @callee_stack16([8 x i64] undef, i64 42, i64 2)

; COMMON: stp {{x[0-9]+}}, {{x[0-9]+}}, [sp]
; COMMON-NEXT: b callee_stack16
  ret void
}


define tailcc void @caller_to8_from24([8 x i64], i64 %a, i64 %b, i64 %c) #0 {
; COMMON-LABEL: caller_to8_from24:
; COMMON-NOT: sub sp,

; Key point is that the "%a" should go where at #16 above SP on entry.
  tail call tailcc void @callee_stack8([8 x i64] undef, i64 42)
  ret void

; COMMON: str {{x[0-9]+}}, [sp, #16]!
; COMMON-NEXT: .cfi_def_cfa_offset -16
; COMMON-NEXT: b callee_stack8
}


define tailcc void @caller_to16_from16([8 x i64], i64 %a, i64 %b) #0 {
; COMMON-LABEL: caller_to16_from16:
; COMMON-NOT: sub sp,

; Here we want to make sure that both loads happen before the stores:
; otherwise either %a or %b will be wrongly clobbered.
  tail call tailcc void @callee_stack16([8 x i64] undef, i64 %b, i64 %a)
  ret void

; COMMON: ldp {{x[0-9]+}}, {{x[0-9]+}}, [sp]
; COMMON: stp {{x[0-9]+}}, {{x[0-9]+}}, [sp]
; COMMON-NEXT: b callee_stack16
}

define tailcc void @disable_tail_calls() nounwind "disable-tail-calls"="true" {
; COMMON-LABEL: disable_tail_calls:
; COMMON-NEXT: // %bb.

  tail call tailcc void @callee_stack0()
  ret void

; COMMON: bl callee_stack0
; COMMON: ret
}

; Weakly-referenced extern functions cannot be tail-called, as AAELF does
; not define the behaviour of branch instructions to undefined weak symbols.
define tailcc void @caller_weak() #0 {
; COMMON-LABEL: caller_weak:
; COMMON: bl callee_weak
  tail call void @callee_weak()
  ret void
}

declare { [2 x float] } @get_vec2()

define { [3 x float] } @test_add_elem() #0 {
; SDAG-LABEL: test_add_elem:
; SDAG: bl get_vec2
; SDAG: fmov s2, #1.0
; SDAG: ret
; GISEL-LABEL: test_add_elem:
; GISEL: str	x30, [sp, #-16]!
; GISEL: bl get_vec2
; GISEL: fmov	s2, #1.0
; GISEL: ldr	x30, [sp], #16
; GISEL: ret

  %call = tail call { [2 x float] } @get_vec2()
  %arr = extractvalue { [2 x float] } %call, 0
  %arr.0 = extractvalue [2 x float] %arr, 0
  %arr.1 = extractvalue [2 x float] %arr, 1

  %res.0 = insertvalue { [3 x float] } undef, float %arr.0, 0, 0
  %res.01 = insertvalue { [3 x float] } %res.0, float %arr.1, 0, 1
  %res.012 = insertvalue { [3 x float] } %res.01, float 1.000000e+00, 0, 2
  ret { [3 x float] } %res.012
}

declare double @get_double()
define { double, [2 x double] } @test_mismatched_insert() #0 {
; COMMON-LABEL: test_mismatched_insert:
; COMMON: bl get_double
; COMMON: bl get_double
; COMMON: bl get_double
; COMMON: ret

  %val0 = call double @get_double()
  %val1 = call double @get_double()
  %val2 = tail call double @get_double()

  %res.0 = insertvalue { double, [2 x double] } undef, double %val0, 0
  %res.01 = insertvalue { double, [2 x double] } %res.0, double %val1, 1, 0
  %res.012 = insertvalue { double, [2 x double] } %res.01, double %val2, 1, 1

  ret { double, [2 x double] } %res.012
}

define void @fromC_totail() #0 {
; COMMON-LABEL: fromC_totail:
; COMMON: sub sp, sp, #32

; COMMON-NOT: sub sp,
; COMMON: mov w[[TMP:[0-9]+]], #42
; COMMON: str x[[TMP]], [sp]
; COMMON: bl callee_stack8
  ; We must reset the stack to where it was before the call by undoing its extra stack pop.
; COMMON: str x[[TMP]], [sp, #-16]!
; COMMON: bl callee_stack8
; COMMON: sub sp, sp, #16

  call tailcc void @callee_stack8([8 x i64] undef, i64 42)
  call tailcc void @callee_stack8([8 x i64] undef, i64 42)
  ret void
}

define void @fromC_totail_noreservedframe(i32 %len) #0 {
; COMMON-LABEL: fromC_totail_noreservedframe:
; COMMON: stp x29, x30, [sp, #-32]!

; COMMON: mov w[[TMP:[0-9]+]], #42
  ; Note stack is subtracted here to allocate space for arg
; COMMON: str x[[TMP]], [sp, #-16]!
; COMMON: bl callee_stack8
  ; And here.
; COMMON: str x[[TMP]], [sp, #-16]!
; COMMON: bl callee_stack8
  ; But not restored here because callee_stack8 did that for us.
; COMMON-NOT: sub sp,

  ; Variable sized allocation prevents reserving frame at start of function so each call must allocate any stack space it needs.
  %var = alloca i32, i32 %len

  call tailcc void @callee_stack8([8 x i64] undef, i64 42)
  call tailcc void @callee_stack8([8 x i64] undef, i64 42)
  ret void
}

declare void @Ccallee_stack8([8 x i64], i64)

define tailcc void @fromtail_toC() #0 {
; COMMON-LABEL: fromtail_toC:
; COMMON: sub sp, sp, #32

; COMMON-NOT: sub sp,
; COMMON: mov w[[TMP:[0-9]+]], #42
; COMMON: str x[[TMP]], [sp]
; COMMON: bl Ccallee_stack8
  ; C callees will return with the stack exactly where we left it, so we mustn't try to fix anything.
; COMMON-NOT: add sp,
; COMMON-NOT: sub sp,
; COMMON: str x[[TMP]], [sp]{{$}}
; COMMON: bl Ccallee_stack8
; COMMON-NOT: sub sp,


  call void @Ccallee_stack8([8 x i64] undef, i64 42)
  call void @Ccallee_stack8([8 x i64] undef, i64 42)
  ret void
}

declare tailcc i32 @all_registers_callee(i32 %p1, i32 %p2, i32 %p3, i32 %p4, i32 %p5, i32 %p6, i32 %p7, i32 %p8, i32 %a, i32 %b)

define tailcc i32 @all_registers_caller(i32 %p1, i32 %p2, i32 %p3, i32 %p4, i32 %p5, i32 %p6, i32 %p7, i32 %p8, i32 %in1, i32 %in2) {
; COMMON-LABEL: all_registers_caller:
; COMMON:       // %bb.0: // %entry
; COMMON-NEXT:    ldr w8, [sp]
; COMMON-NEXT:    ldr w9, [sp, #8]
; COMMON-NEXT:    add w8, w8, w0
; COMMON-NEXT:    str w9, [sp]
; COMMON-NEXT:    str w8, [sp, #8]
; COMMON-NEXT:    b all_registers_callee
entry:
  %tmp = add i32 %in1, %p1
  %retval = tail call tailcc i32 @all_registers_callee(i32 %p1, i32 %p2, i32 %p3, i32 %p4, i32 %p5, i32 %p6, i32 %p7, i32 %p8, i32 %in2, i32 %tmp)
  ret i32 %retval
}

define tailcc noundef i64 @call_with_byval_caller(i64 noundef %a, i64 noundef %d) {
; COMMON-LABEL: call_with_byval_caller:
; COMMON:       // %bb.0: // %start
; COMMON-NEXT:    mov x8, #-4919131752989213765 // =0xbbbbbbbbbbbbbbbb
; COMMON-NEXT:    stp x0, x8, [sp, #-64]!
; COMMON-NEXT:    .cfi_def_cfa_offset 64
; COMMON-NEXT:    mov x9, #-3689348814741910324 // =0xcccccccccccccccc
; COMMON-NEXT:    stp x9, x1, [sp, #16]
; COMMON-NEXT:    ldp q1, q0, [sp]
; COMMON-NEXT:    stp q1, q0, [sp, #32]!
; COMMON-NEXT:    b call_with_byval_callee
start:
  %large = alloca [4 x i64], align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %large)
  store i64 %a, ptr %large, align 8
  %0 = getelementptr inbounds nuw i8, ptr %large, i64 8
  store i64 -4919131752989213765, ptr %0, align 8
  %1 = getelementptr inbounds nuw i8, ptr %large, i64 16
  store i64 -3689348814741910324, ptr %1, align 8
  %2 = getelementptr inbounds nuw i8, ptr %large, i64 24
  store i64 %d, ptr %2, align 8
  %3 = musttail call tailcc i64 @call_with_byval_callee(ptr byval([4 x i64]) %large)
  ret i64 %3
}

declare tailcc noundef i64 @call_with_byval_callee(ptr byval([4 x i64]) %large)

@array_32xi8 = constant [32 x i8] c"\01\00\00\00\00\00\00\00\02\00\00\00\00\00\00\00\03\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00"
define tailcc i64 @from_global() {
; COMMON-LABEL: from_global:
; COMMON:       // %bb.0:
; COMMON-NEXT:    adrp x8, :got:array_32xi8
; COMMON-NEXT:    ldr x8, [x8, :got_lo12:array_32xi8]
; COMMON-NEXT:    ldr q0, [x8]
; COMMON-NEXT:    str q0, [sp, #-32]!
; COMMON-NEXT:    .cfi_def_cfa_offset 32
; COMMON-NEXT:    ldr q0, [x8, #16]
; COMMON-NEXT:    str q0, [sp, #16]
; COMMON-NEXT:    b callee_byval_32xi8
  %r = musttail call tailcc i64 @callee_byval_32xi8(ptr byval([32 x i8]) align 8 @array_32xi8)
  ret i64 %r
}

declare tailcc i64 @callee_byval_32xi8(ptr byval([32 x i8]) align 8)

define tailcc i64 @forward_incoming(ptr byval([32 x i8]) align 8 %p) {
; COMMON-LABEL: forward_incoming:
; COMMON:       // %bb.0:
; COMMON-NEXT:    b callee_byval_32xi8
  %r = tail call tailcc i64 @callee_byval_32xi8(ptr byval([32 x i8]) align 8 %p)
  ret i64 %r
}

define tailcc i64 @swap_incoming(ptr byval([32 x i8]) align 8 %p, ptr byval([32 x i8]) align 8 %q) {
; SDAG-LABEL: swap_incoming:
; SDAG:       // %bb.0:
; SDAG-NEXT:    sub sp, sp, #64
; SDAG-NEXT:    .cfi_def_cfa_offset 64
; SDAG-NEXT:    ldp q1, q0, [sp, #96]
; SDAG-NEXT:    stp q1, q0, [sp, #32]
; SDAG-NEXT:    ldp q1, q0, [sp, #64]
; SDAG-NEXT:    stp q1, q0, [sp]
; SDAG-NEXT:    ldp q1, q0, [sp, #32]
; SDAG-NEXT:    stp q1, q0, [sp, #64]
; SDAG-NEXT:    ldp q1, q0, [sp]
; SDAG-NEXT:    stp q1, q0, [sp, #96]
; SDAG-NEXT:    add sp, sp, #64
; SDAG-NEXT:    b swap_incoming_callee
;
; GISEL-LABEL: swap_incoming:
; GISEL:       // %bb.0:
; GISEL-NEXT:    sub sp, sp, #64
; GISEL-NEXT:    .cfi_def_cfa_offset 64
; GISEL-NEXT:    ldp q1, q0, [sp, #96]
; GISEL-NEXT:    stp q1, q0, [sp, #32]
; GISEL-NEXT:    ldp q1, q0, [sp, #32]
; GISEL-NEXT:    stp q1, q0, [sp, #64]
; GISEL-NEXT:    ldp q1, q0, [sp, #64]
; GISEL-NEXT:    stp q1, q0, [sp]
; GISEL-NEXT:    ldp q1, q0, [sp]
; GISEL-NEXT:    stp q1, q0, [sp, #96]
; GISEL-NEXT:    add sp, sp, #64
; GISEL-NEXT:    b swap_incoming_callee
  %r = tail call tailcc i64 @swap_incoming_callee(ptr byval([32 x i8]) align 8 %q, ptr byval([32 x i8]) align 8 %p)
  ret i64 %r
}

declare tailcc i64 @swap_incoming_callee(ptr byval([32 x i8]) align 8, ptr byval([32 x i8]) align 8)

define tailcc noundef i64 @swap_local_byval(i64 noundef %a, i64 noundef %d, ptr byval([32 x i8]) align 8 %p) {
; COMMON-LABEL: swap_local_byval:
; COMMON:       // %bb.0: // %start
; COMMON-NEXT:    mov x8, #-4919131752989213765 // =0xbbbbbbbbbbbbbbbb
; COMMON-NEXT:    stp x0, x8, [sp, #-64]!
; COMMON-NEXT:    .cfi_def_cfa_offset 64
; COMMON-NEXT:    mov x9, #-3689348814741910324 // =0xcccccccccccccccc
; COMMON-NEXT:    stp x9, x1, [sp, #16]
; COMMON-NEXT:    ldp q1, q0, [sp]
; COMMON-NEXT:    stp q1, q0, [sp, #32]!
; COMMON-NEXT:    b swap_incoming_callee
start:
  %large = alloca [32 x i8], align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %large)
  store i64 %a, ptr %large, align 8
  %0 = getelementptr inbounds nuw i8, ptr %large, i64 8
  store i64 -4919131752989213765, ptr %0, align 8
  %1 = getelementptr inbounds nuw i8, ptr %large, i64 16
  store i64 -3689348814741910324, ptr %1, align 8
  %2 = getelementptr inbounds nuw i8, ptr %large, i64 24
  store i64 %d, ptr %2, align 8
  %3 = musttail call tailcc i64 @swap_incoming_callee(ptr byval([32 x i8]) align 8 %large, ptr byval([32 x i8]) align 8 %p)
  ret i64 %3
}

declare tailcc i64 @overlap_callee(i64, i64, i64, i64, i64, i64, i64, i64, i64, ptr byval([64 x i8]) align 8)

define tailcc i64 @forward_overlap(i64 %r0, i64 %r1, i64 %r2, i64 %r3, i64 %r4, i64 %r5, i64 %r6, i64 %r7, ptr byval([64 x i8]) align 8 %p) {
; COMMON-LABEL: forward_overlap:
; COMMON:       // %bb.0:
; COMMON-NEXT:    sub sp, sp, #80
; COMMON-NEXT:    .cfi_def_cfa_offset 80
; COMMON-NEXT:    ldp q1, q0, [sp, #80]
; COMMON-NEXT:    mov w8, #99 // =0x63
; COMMON-NEXT:    str x8, [sp, #64]
; COMMON-NEXT:    stp q1, q0, [sp]
; COMMON-NEXT:    ldp q1, q0, [sp, #112]
; COMMON-NEXT:    stp q1, q0, [sp, #32]
; COMMON-NEXT:    ldp q1, q0, [sp]
; COMMON-NEXT:    stur q1, [sp, #72]
; COMMON-NEXT:    stur q0, [sp, #88]
; COMMON-NEXT:    ldp q1, q0, [sp, #32]
; COMMON-NEXT:    stur q1, [sp, #104]
; COMMON-NEXT:    stur q0, [sp, #120]
; COMMON-NEXT:    add sp, sp, #64
; COMMON-NEXT:    b overlap_callee
  %r = musttail call tailcc i64 @overlap_callee(i64 %r0, i64 %r1, i64 %r2, i64 %r3, i64 %r4, i64 %r5, i64 %r6, i64 %r7, i64 99, ptr byval([64 x i8]) align 8 %p)
  ret i64 %r
}

attributes #0 = { uwtable }
