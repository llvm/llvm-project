; RUN: llc < %s -mtriple=thumbv7-apple-darwin -mcpu=cortex-a8 \
; RUN:   -pre-RA-sched=source | FileCheck %s
; RUN: llc < %s -mtriple=thumbv7-apple-darwin -mcpu=cortex-a8 \
; RUN:   -pre-RA-sched=list-hybrid | FileCheck %s
; RUN: llc < %s -mtriple=thumbv7-apple-darwin -mcpu=cortex-a8 -regalloc=basic | FileCheck %s
; Radar 7459078
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:64:64-v128:128:128-a0:0:32-n32"
	
%0 = type { i32, i32 }
%s1 = type { %s3, i32, %s4, ptr, ptr, ptr, ptr, ptr, ptr, i32, i64, [1 x i32] }
%s2 = type { ptr, %s4 }
%s3 = type { %s2, i32, i32, ptr, [4 x i8], float, %s4, ptr, ptr }
%s4 = type { %s5 }
%s5 = type { i32 }

; Make sure the cmp is not scheduled before the InlineAsm that clobbers cc.
; CHECK: bl _f2
; CHECK: clz {{r[0-9]+}}
; CHECK-DAG: lsrs    {{r[0-9]+}}
; CHECK-DAG: lsls    {{r[0-9]+}}
; CHECK-NEXT: orr.w   {{r[0-9]+}}
; CHECK-NEXT: InlineAsm Start
define void @test(ptr %this, i32 %format, i32 %w, i32 %h, i32 %levels, ptr %s, ptr %data, ptr nocapture %rowbytes, ptr %release, ptr %info) nounwind {
entry:
  %tmp1 = getelementptr inbounds %s1, ptr %this, i32 0, i32 0, i32 0, i32 1, i32 0, i32 0
  store volatile i32 1, ptr %tmp1, align 4
  %tmp12 = getelementptr inbounds %s1, ptr %this, i32 0, i32 1
  store i32 %levels, ptr %tmp12, align 4
  %tmp13 = getelementptr inbounds %s1, ptr %this, i32 0, i32 3
  store ptr %data, ptr %tmp13, align 4
  %tmp14 = getelementptr inbounds %s1, ptr %this, i32 0, i32 4
  store ptr %release, ptr %tmp14, align 4
  %tmp15 = getelementptr inbounds %s1, ptr %this, i32 0, i32 5
  store ptr %info, ptr %tmp15, align 4
  %tmp16 = getelementptr inbounds %s1, ptr %this, i32 0, i32 6
  store ptr null, ptr %tmp16, align 4
  %tmp17 = getelementptr inbounds %s1, ptr %this, i32 0, i32 7
  store ptr null, ptr %tmp17, align 4
  %tmp19 = getelementptr inbounds %s1, ptr %this, i32 0, i32 10
  store i64 0, ptr %tmp19, align 4
  tail call  void @f1(ptr %this, ptr %s) nounwind
  %tmp21 = shl i32 %format, 6
  %tmp22 = tail call  zeroext i8 @f2(i32 %format) nounwind
  %toBoolnot = icmp eq i8 %tmp22, 0
  %tmp23 = zext i1 %toBoolnot to i32
  %flags.0 = or i32 %tmp23, %tmp21
  %tmp24 = shl i32 %flags.0, 16
  %asmtmp.i.i.i = tail call %0 asm sideeffect "\0A0:\09ldrex $1, [$2]\0A\09orr $1, $1, $3\0A\09strex $0, $1, [$2]\0A\09cmp $0, #0\0A\09bne 0b", "=&r,=&r,r,r,~{memory},~{cc}"(ptr %tmp1, i32 %tmp24) nounwind
  %tmp25 = getelementptr inbounds %s1, ptr %this, i32 0, i32 2, i32 0, i32 0
  store volatile i32 1, ptr %tmp25, align 4
  %tmp26 = icmp eq i32 %levels, 0
  br i1 %tmp26, label %return, label %bb4

bb4:
  %l.09 = phi i32 [ %tmp28, %bb4 ], [ 0, %entry ]
  %scevgep = getelementptr %s1, ptr %this, i32 0, i32 11, i32 %l.09
  %scevgep10 = getelementptr i32, ptr %rowbytes, i32 %l.09
  %tmp27 = load i32, ptr %scevgep10, align 4
  store i32 %tmp27, ptr %scevgep, align 4
  %tmp28 = add i32 %l.09, 1
  %exitcond = icmp eq i32 %tmp28, %levels
  br i1 %exitcond, label %return, label %bb4

return:
  ret void
}

declare void @f1(ptr, ptr)
declare zeroext i8 @f2(i32)
