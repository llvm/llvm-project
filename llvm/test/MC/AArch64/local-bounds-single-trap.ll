; RUN: llc -O3 -mtriple arm64-linux -filetype asm -o - %s | FileCheck %s -check-prefix CHECK-ASM
; This test checks that nomerge correctly prevents the traps from being merged
; in the compiled code.
; Prior to ae6dc64ec670891cb15049277e43133d4df7fb4b, this test showed that
; nomerge did not work correctly.

@B = dso_local global [10 x i8] zeroinitializer, align 1
@B2 = dso_local global [10 x i8] zeroinitializer, align 1

; Function Attrs: noinline nounwind uwtable
define dso_local void @f8(i32 noundef %i, i32 noundef %k) #0 {
entry:
; CHECK-ASM: 	cmp	x8, #10
; CHECK-ASM: 	b.hi	.LBB0_5
; CHECK-ASM: // %bb.1:                               // %entry
; CHECK-ASM: 	mov	w9, #10                         // =0xa
; CHECK-ASM: 	sub	x9, x9, x8
; CHECK-ASM: 	cbz	x9, .LBB0_5
; CHECK-ASM: // %bb.2:
; CHECK-ASM: 	ldrsw	x9, [sp, #8]
; CHECK-ASM: 	adrp	x10, B
; CHECK-ASM: 	add	x10, x10, :lo12:B
; CHECK-ASM: 	strb	wzr, [x10, x8]
; CHECK-ASM: 	cmp	x9, #10
; CHECK-ASM: 	b.hi	.LBB0_6
; CHECK-ASM: // %bb.3:
; CHECK-ASM: 	mov	w8, #10                         // =0xa
; CHECK-ASM: 	sub	x8, x8, x9
; CHECK-ASM: 	cbz	x8, .LBB0_6
; CHECK-ASM: // %bb.4:
; CHECK-ASM: 	adrp	x8, B2
; CHECK-ASM: 	add	x8, x8, :lo12:B2
; CHECK-ASM: 	strb	wzr, [x8, x9]
; CHECK-ASM: 	add	sp, sp, #16
; CHECK-ASM: 	.cfi_def_cfa_offset 0
; CHECK-ASM: 	ret
; CHECK-ASM: .LBB0_5:                                // %trap
; CHECK-ASM: .cfi_restore_state
; CHECK-ASM: brk     #0x1
; CHECK-ASM: .LBB0_6:                                // %trap3
; CHECK-ASM: brk     #0x1
  %i.addr = alloca i32, align 4
  %k.addr = alloca i32, align 4
  store i32 %i, ptr %i.addr, align 4
  store i32 %k, ptr %k.addr, align 4
  %0 = load i32, ptr %i.addr, align 4
  %idxprom = sext i32 %0 to i64
  %1 = add i64 0, %idxprom
  %arrayidx = getelementptr inbounds [10 x i8], ptr @B, i64 0, i64 %idxprom
  %2 = sub i64 10, %1
  %3 = icmp ult i64 10, %1
  %4 = icmp ult i64 %2, 1
  %5 = or i1 %3, %4
  br i1 %5, label %trap, label %6

6:                                                ; preds = %entry
  store i8 0, ptr %arrayidx, align 1
  %7 = load i32, ptr %k.addr, align 4
  %idxprom1 = sext i32 %7 to i64
  %8 = add i64 0, %idxprom1
  %arrayidx2 = getelementptr inbounds [10 x i8], ptr @B2, i64 0, i64 %idxprom1
  %9 = sub i64 10, %8
  %10 = icmp ult i64 10, %8
  %11 = icmp ult i64 %9, 1
  %12 = or i1 %10, %11
  br i1 %12, label %trap3, label %13

13:                                               ; preds = %6
  store i8 0, ptr %arrayidx2, align 1
  ret void

trap:                                             ; preds = %entry
  call void @llvm.trap() #2
  unreachable

trap3:                                            ; preds = %6
  call void @llvm.trap() #2
  unreachable
}

; Function Attrs: cold noreturn nounwind memory(inaccessiblemem: write)
declare void @llvm.trap() #1

attributes #0 = { noinline nounwind uwtable }
attributes #1 = { cold noreturn nounwind memory(inaccessiblemem: write) }
attributes #2 = { noreturn nounwind nomerge }
