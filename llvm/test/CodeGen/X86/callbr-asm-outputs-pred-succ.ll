; Tests that InstrEmitter::EmitMachineNode correctly sets predecessors and
; successors.

; RUN: llc -stop-after=finalize-isel -print-after=finalize-isel -mtriple=i686-- < %s 2>&1 | FileCheck %s

; The block containting the INLINEASM_BR should have a fallthrough and its
; indirect targets as its successors. Fallthrough should have 100% branch weight,
; while the indirect targets have 0%.
; CHECK: bb.0 (%ir-block.2):
; CHECK-NEXT: successors: %bb.1(0x80000000), %bb.4(0x00000000); %bb.1(100.00%), %bb.4(0.00%)

; The fallthrough is a block containing a second INLINEASM_BR. Check it has two successors,
; and the the probability for fallthrough is 100%.
; CHECK: bb.1 (%ir-block.4):
; CHECK-NEXT: predecessors: %bb.0
; CHECK-NEXT: successors: %bb.3(0x80000000), %bb.2(0x00000000); %bb.3(100.00%), %bb.2(0.00%)

; Check the second INLINEASM_BR target block is preceded by the block with the
; second INLINEASM_BR.
; CHECK: bb.2 (%ir-block.7, address-taken, inlineasm-br-indirect-target):
; CHECK-NEXT: predecessors: %bb.1

; Check the first INLINEASM_BR target block is predecessed by the block with
; the first INLINEASM_BR.
; CHECK: bb.4 (%ir-block.11, address-taken, inlineasm-br-indirect-target):
; CHECK-NEXT: predecessors: %bb.0

@.str = private unnamed_addr constant [26 x i8] c"inline asm#1 returned %d\0A\00", align 1
@.str.2 = private unnamed_addr constant [26 x i8] c"inline asm#2 returned %d\0A\00", align 1
@str = private unnamed_addr constant [30 x i8] c"inline asm#1 caused exception\00", align 1
@str.4 = private unnamed_addr constant [30 x i8] c"inline asm#2 caused exception\00", align 1

; Function Attrs: nounwind uwtable
define dso_local i32 @main(i32 %0, ptr nocapture readnone %1) #0 {
  %3 = callbr i32 asm "jmp ${1:l}", "=r,i,~{dirflag},~{fpsr},~{flags}"(ptr blockaddress(@main, %11)) #3
          to label %4 [label %11]

4:                                                ; preds = %2
  %5 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @.str, i32 %3)
  %6 = callbr i32 asm "jmp ${1:l}", "=r,i,~{dirflag},~{fpsr},~{flags}"(ptr blockaddress(@main, %7)) #3
          to label %9 [label %7]

7:                                                ; preds = %4
  %8 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.4)
  br label %13

9:                                                ; preds = %4
  %10 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @.str.2, i32 %6)
  br label %13

11:                                               ; preds = %2
  %12 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str)
  br label %13

13:                                               ; preds = %11, %9, %7
  %14 = phi i32 [ 1, %7 ], [ 0, %9 ], [ 1, %11 ]
  ret i32 %14
}

declare dso_local i32 @printf(ptr nocapture readonly, ...) local_unnamed_addr #1
declare i32 @puts(ptr nocapture readonly) local_unnamed_addr #2
