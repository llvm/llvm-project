; RUN: llc -mtriple=aarch64-none-linux-gnu -max-bytes-for-alignment=8 --align-loops=32 < %s -o -| FileCheck %s --check-prefixes=CHECK,CHECK-EXPLICIT
; RUN: llc -mtriple=aarch64-none-linux-gnu --align-loops=32 < %s -o -| FileCheck %s --check-prefixes=CHECK,CHECK-IMPLICIT
; RUN: llc -mtriple=aarch64-none-linux-gnu --align-loops=32 < %s -o - --filetype=obj | llvm-objdump --arch=aarch64  -d -| FileCheck %s --check-prefixes=CHECK-OBJ,CHECK-OBJ-IMPLICIT
; RUN: llc -mtriple=aarch64-none-linux-gnu -max-bytes-for-alignment=8 --align-loops=32 < %s -o - --filetype=obj | llvm-objdump --arch=aarch64  -d -| FileCheck %s --check-prefixes=CHECK-OBJ,CHECK-OBJ-EXPLICIT

; This test is checking that the correct operands to the .p2align are emitted correctly, and that the resulting obj
; is padded as expected. The key interest in the CHECK-OBJ-* sections is the size of the padding region (the nops),
; and not the exact instructions either side of them (But the last instruction of the EXPLICIT and IMPLICIT checks
; should be the same, at different locations)
define i32 @a(i32 %x, i32* nocapture readonly %y, i32* nocapture readonly %z) {
; CHECK-LABEL: a:
; CHECK-EXPLICIT:    .p2align 5, 0x0, 8
; CHECK-IMPLICIT:    .p2align 5
; CHECK-NEXT:  .LBB0_5: // %vector.body
; CHECK-EXPLICIT:    .p2align 5, 0x0, 8
; CHECK-IMPLICIT:    .p2align 5
; CHECK-NEXT:  .LBB0_8: // %for.body
; CHECK-OBJ;Disassembly of section .text:
; CHECK-OBJ:               88: 8b0a002a      add
; CHECK-OBJ-IMPLICIT-NEXT: 8c: d503201f      nop
; CHECK-OBJ-IMPLICIT-NEXT: 90: d503201f      nop
; CHECK-OBJ-IMPLICIT-NEXT: 94: d503201f      nop
; CHECK-OBJ-IMPLICIT-NEXT: 98: d503201f      nop
; CHECK-OBJ-IMPLICIT-NEXT: 9c: d503201f      nop
; CHECK-OBJ-IMPLICIT-NEXT: a0: b840454b      ldr
; CHECK-OBJ-EXPLICIT-NEXT: 8c: b840454b      ldr
entry:
  %cmp10 = icmp sgt i32 %x, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %x to i64
  %min.iters.check = icmp ult i32 %x, 8
  br i1 %min.iters.check, label %for.body.preheader17, label %vector.ph

vector.ph:                                        ; preds = %for.body.preheader
  %n.vec = and i64 %wide.trip.count, 4294967288
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %vec.phi = phi <4 x i32> [ zeroinitializer, %vector.ph ], [ %10, %vector.body ]
  %vec.phi13 = phi <4 x i32> [ zeroinitializer, %vector.ph ], [ %11, %vector.body ]
  %0 = getelementptr inbounds i32, i32* %y, i64 %index
  %1 = bitcast i32* %0 to <4 x i32>*
  %wide.load = load <4 x i32>, <4 x i32>* %1, align 4
  %2 = getelementptr inbounds i32, i32* %0, i64 4
  %3 = bitcast i32* %2 to <4 x i32>*
  %wide.load14 = load <4 x i32>, <4 x i32>* %3, align 4
  %4 = getelementptr inbounds i32, i32* %z, i64 %index
  %5 = bitcast i32* %4 to <4 x i32>*
  %wide.load15 = load <4 x i32>, <4 x i32>* %5, align 4
  %6 = getelementptr inbounds i32, i32* %4, i64 4
  %7 = bitcast i32* %6 to <4 x i32>*
  %wide.load16 = load <4 x i32>, <4 x i32>* %7, align 4
  %8 = add <4 x i32> %wide.load, %vec.phi
  %9 = add <4 x i32> %wide.load14, %vec.phi13
  %10 = add <4 x i32> %8, %wide.load15
  %11 = add <4 x i32> %9, %wide.load16
  %index.next = add nuw i64 %index, 8
  %12 = icmp eq i64 %index.next, %n.vec
  br i1 %12, label %middle.block, label %vector.body

middle.block:                                     ; preds = %vector.body
  %bin.rdx = add <4 x i32> %11, %10
  %13 = call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> %bin.rdx)
  %cmp.n = icmp eq i64 %n.vec, %wide.trip.count
  br i1 %cmp.n, label %for.cond.cleanup, label %for.body.preheader17

for.body.preheader17:                             ; preds = %for.body.preheader, %middle.block
  %indvars.iv.ph = phi i64 [ 0, %for.body.preheader ], [ %n.vec, %middle.block ]
  %b.011.ph = phi i32 [ 0, %for.body.preheader ], [ %13, %middle.block ]
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %middle.block, %entry
  %b.0.lcssa = phi i32 [ 0, %entry ], [ %13, %middle.block ], [ %add3, %for.body ]
  ret i32 %b.0.lcssa

for.body:                                         ; preds = %for.body.preheader17, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ %indvars.iv.ph, %for.body.preheader17 ]
  %b.011 = phi i32 [ %add3, %for.body ], [ %b.011.ph, %for.body.preheader17 ]
  %arrayidx = getelementptr inbounds i32, i32* %y, i64 %indvars.iv
  %14 = load i32, i32* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, i32* %z, i64 %indvars.iv
  %15 = load i32, i32* %arrayidx2, align 4
  %add = add i32 %14, %b.011
  %add3 = add i32 %add, %15
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

declare i32 @llvm.vector.reduce.add.v4i32(<4 x i32>)
