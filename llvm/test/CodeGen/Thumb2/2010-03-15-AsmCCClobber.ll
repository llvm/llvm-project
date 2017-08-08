; RUN: llc < %s -mtriple=thumbv7-apple-darwin -mcpu=cortex-a8 \
; RUN:   -pre-RA-sched=source | FileCheck %s
; RUN: llc < %s -mtriple=thumbv7-apple-darwin -mcpu=cortex-a8 \
; RUN:   -pre-RA-sched=list-hybrid | FileCheck %s
; RUN: llc < %s -mtriple=thumbv7-apple-darwin -mcpu=cortex-a8 -regalloc=basic | FileCheck %s
; Radar 7459078
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:64:64-v128:128:128-a0:0:32-n32"
	
%0 = type { i32, i32 }
%s1 = type { %s3, i32, %s4, i8*, void (i8*, i8*)*, i8*, i32*, i32*, i32*, i32, i64, [1 x i32] }
%s2 = type { i32 (...)**, %s4 }
%s3 = type { %s2, i32, i32, i32*, [4 x i8], float, %s4, i8*, i8* }
%s4 = type { %s5 }
%s5 = type { i32 }

; Make sure the cmp is not scheduled before the InlineAsm that clobbers cc.
; CHECK: bl _f2
; CHECK: cmp r0, #0
; CHECK-NOT: cmp
; CHECK: InlineAsm Start
define void @test(%s1* %this, i32 %format, i32 %w, i32 %h, i32 %levels, i32* %s, i8* %data, i32* nocapture %rowbytes, void (i8*, i8*)* %release, i8* %info) nounwind {
entry:
  %tmp1 = getelementptr inbounds %s1, %s1* %this, i32 0, i32 0, i32 0, i32 1, i32 0, i32 0
  store volatile i32 1, i32* %tmp1, align 4
  %tmp12 = getelementptr inbounds %s1, %s1* %this, i32 0, i32 1
  store i32 %levels, i32* %tmp12, align 4
  %tmp13 = getelementptr inbounds %s1, %s1* %this, i32 0, i32 3
  store i8* %data, i8** %tmp13, align 4
  %tmp14 = getelementptr inbounds %s1, %s1* %this, i32 0, i32 4
  store void (i8*, i8*)* %release, void (i8*, i8*)** %tmp14, align 4
  %tmp15 = getelementptr inbounds %s1, %s1* %this, i32 0, i32 5
  store i8* %info, i8** %tmp15, align 4
  %tmp16 = getelementptr inbounds %s1, %s1* %this, i32 0, i32 6
  store i32* null, i32** %tmp16, align 4
  %tmp17 = getelementptr inbounds %s1, %s1* %this, i32 0, i32 7
  store i32* null, i32** %tmp17, align 4
  %tmp19 = getelementptr inbounds %s1, %s1* %this, i32 0, i32 10
  store i64 0, i64* %tmp19, align 4
  %tmp20 = getelementptr inbounds %s1, %s1* %this, i32 0, i32 0
  tail call  void @f1(%s3* %tmp20, i32* %s) nounwind
  %tmp21 = shl i32 %format, 6
  %tmp22 = tail call  zeroext i8 @f2(i32 %format) nounwind
  %toBoolnot = icmp eq i8 %tmp22, 0
  %tmp23 = zext i1 %toBoolnot to i32
  %flags.0 = or i32 %tmp23, %tmp21
  %tmp24 = shl i32 %flags.0, 16
  %asmtmp.i.i.i = tail call %0 asm sideeffect "\0A0:\09ldrex $1, [$2]\0A\09orr $1, $1, $3\0A\09strex $0, $1, [$2]\0A\09cmp $0, #0\0A\09bne 0b", "=&r,=&r,r,r,~{memory},~{cc}"(i32* %tmp1, i32 %tmp24) nounwind
  %tmp25 = getelementptr inbounds %s1, %s1* %this, i32 0, i32 2, i32 0, i32 0
  store volatile i32 1, i32* %tmp25, align 4
  %tmp26 = icmp eq i32 %levels, 0
  br i1 %tmp26, label %return, label %bb4

bb4:
  %l.09 = phi i32 [ %tmp28, %bb4 ], [ 0, %entry ]
  %scevgep = getelementptr %s1, %s1* %this, i32 0, i32 11, i32 %l.09
  %scevgep10 = getelementptr i32, i32* %rowbytes, i32 %l.09
  %tmp27 = load i32, i32* %scevgep10, align 4
  store i32 %tmp27, i32* %scevgep, align 4
  %tmp28 = add i32 %l.09, 1
  %exitcond = icmp eq i32 %tmp28, %levels
  br i1 %exitcond, label %return, label %bb4

return:
  ret void
}

declare void @f1(%s3*, i32*)
declare zeroext i8 @f2(i32)
