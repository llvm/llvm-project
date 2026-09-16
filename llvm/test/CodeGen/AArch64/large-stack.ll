; RUN: llc < %s -mtriple=aarch64-none-linux-gnu | FileCheck %s

@.str = private unnamed_addr constant [11 x i8] c"val = %ld\0A\00", align 1

; Function Attrs: noinline optnone
define dso_local void @set_large(i64 %val) #0 {
entry:
  %val.addr = alloca i64, align 8
  %large = alloca [268435456 x i64], align 8
  %i = alloca i32, align 4
  store i64 %val, ptr %val.addr, align 8
  %0 = load i64, ptr %val.addr, align 8
  %arrayidx = getelementptr inbounds [268435456 x i64], ptr %large, i64 0, i64 %0
  store i64 1, ptr %arrayidx, align 8
  %1 = load i64, ptr %val.addr, align 8
  %arrayidx1 = getelementptr inbounds [268435456 x i64], ptr %large, i64 0, i64 %1
  %2 = load i64, ptr %arrayidx1, align 8
  %call = call i32 (ptr, ...) @printf(ptr @.str, i64 %2)
  ret void
}

declare dso_local i32 @printf(ptr, ...)

attributes #0 = { noinline optnone "frame-pointer"="all" uwtable }

; CHECK:                  stp	x[[SPILL_REG1:[0-9]+]], x[[SPILL_REG2:[0-9]+]], [sp, #-[[SPILL_OFFSET1:[0-9]+]]]
; CHECK-NEXT:             .cfi_def_cfa_offset [[SPILL_OFFSET1]]
; CHECK-NEXT:             str	x[[SPILL_REG3:[0-9]+]], [sp, #[[SPILL_OFFSET2:[0-9]+]]]
; CHECK-NEXT:             mov	x[[FRAME:[0-9]+]], sp
; CHECK-NEXT:             .cfi_def_cfa w[[FRAME]], [[SPILL_OFFSET1]]
; CHECK-COUNT-128:        sub	sp, sp, #[[STACK1:[0-9]+]], lsl #12
; CHECK-NEXT:             sub	sp, sp, #[[STACK2:[0-9]+]], lsl #12
; CHECK-NEXT:             sub	sp, sp, #[[STACK3:[0-9]+]]
; CHECK:                  sub	x[[INDEX:[0-9]+]], x[[FRAME]], #8
; CHECK-NEXT:             str	x0, [x[[INDEX]]]
; CHECK-NEXT:             ldr	x[[VAL1:[0-9]+]], [x[[INDEX]]]
; CHECK-NEXT:             add x10, sp, #8
; CHECK-NEXT:             mov x11, #1 // =0x1
; CHECK-NEXT:             str x11, [x10, x9, lsl #3]
; CHECK-NEXT:             ldr x8, [x8]
; CHECK-NEXT:             ldr x1, [x10, x8, lsl #3]
; CHECK-NEXT:             adrp x0, .L.str
; CHECK-NEXT:             add x0, x0, :lo12:.L.str
; CHECK:                  bl	printf
; CHECK-COUNT-128:        add	sp, sp, #[[STACK1]], lsl #12
; CHECK-NEXT:             add	sp, sp, #[[STACK2]], lsl #12
; CHECK-NEXT:             add	sp, sp, #[[STACK3]]
; CHECK-NEXT:	            .cfi_def_cfa wsp, [[SPILL_OFFSET1]]
; CHECK-NEXT:             ldr	x[[SPILL_REG3]], [sp, #[[SPILL_OFFSET2]]]
; CHECK-NEXT:             ldp	x[[SPILL_REG1]], x[[SPILL_REG2]], [sp], #[[SPILL_OFFSET1]]
; CHECK-NEXT:           	.cfi_def_cfa_offset 0
; CHECK-NEXT:           	.cfi_restore w[[SPILL_REG3]]
; CHECK-NEXT:           	.cfi_restore w[[SPILL_REG2]]
; CHECK-NEXT:           	.cfi_restore w[[SPILL_REG1]]
