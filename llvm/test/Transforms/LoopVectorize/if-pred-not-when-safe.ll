; RUN: opt -S -force-vector-width=2 -force-vector-interleave=1 -passes=loop-vectorize,simplifycfg -verify-loop-info -simplifycfg-require-and-preserve-domtree=1 < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Test no-predication of instructions that are provably safe, e.g. dividing by
; a non-zero constant.
define void @test(ptr nocapture %asd, ptr nocapture %aud,
                  ptr nocapture %asr, ptr nocapture %aur,
                  ptr nocapture %asd0, ptr nocapture %aud0,
                  ptr nocapture %asr0, ptr nocapture %aur0
) {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %if.end
  ret void

; CHECK-LABEL: test
; CHECK: vector.body:
; CHECK: %{{.*}} = sdiv <2 x i32> %{{.*}}, splat (i32 11)
; CHECK: %{{.*}} = udiv <2 x i32> %{{.*}}, splat (i32 13)
; CHECK: %{{.*}} = srem <2 x i32> %{{.*}}, splat (i32 17)
; CHECK: %{{.*}} = urem <2 x i32> %{{.*}}, splat (i32 19)
; CHECK-NOT: %{{.*}} = sdiv <2 x i32> %{{.*}}, <i32 0, i32 0>
; CHECK-NOT: %{{.*}} = udiv <2 x i32> %{{.*}}, <i32 0, i32 0>
; CHECK-NOT: %{{.*}} = srem <2 x i32> %{{.*}}, <i32 0, i32 0>
; CHECK-NOT: %{{.*}} = urem <2 x i32> %{{.*}}, <i32 0, i32 0>

for.body:                                         ; preds = %if.end, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %if.end ]
  %isd = getelementptr inbounds i32, ptr %asd, i64 %indvars.iv
  %iud = getelementptr inbounds i32, ptr %aud, i64 %indvars.iv
  %isr = getelementptr inbounds i32, ptr %asr, i64 %indvars.iv
  %iur = getelementptr inbounds i32, ptr %aur, i64 %indvars.iv
  %lsd = load i32, ptr %isd, align 4
  %lud = load i32, ptr %iud, align 4
  %lsr = load i32, ptr %isr, align 4
  %lur = load i32, ptr %iur, align 4
  %psd = add nsw i32 %lsd, 23
  %pud = add nsw i32 %lud, 24
  %psr = add nsw i32 %lsr, 25
  %pur = add nsw i32 %lur, 26
  %isd0 = getelementptr inbounds i32, ptr %asd0, i64 %indvars.iv
  %iud0 = getelementptr inbounds i32, ptr %aud0, i64 %indvars.iv
  %isr0 = getelementptr inbounds i32, ptr %asr0, i64 %indvars.iv
  %iur0 = getelementptr inbounds i32, ptr %aur0, i64 %indvars.iv
  %lsd0 = load i32, ptr %isd0, align 4
  %lud0 = load i32, ptr %iud0, align 4
  %lsr0 = load i32, ptr %isr0, align 4
  %lur0 = load i32, ptr %iur0, align 4
  %psd0 = add nsw i32 %lsd, 27
  %pud0 = add nsw i32 %lud, 28
  %psr0 = add nsw i32 %lsr, 29
  %pur0 = add nsw i32 %lur, 30
  %cmp1 = icmp slt i32 %lsd, 100
  br i1 %cmp1, label %if.then, label %if.end

if.then:                                          ; preds = %for.body
  %rsd = sdiv i32 %psd, 11
  %rud = udiv i32 %pud, 13
  %rsr = srem i32 %psr, 17
  %rur = urem i32 %pur, 19
  %rsd0 = sdiv i32 %psd0, 0
  %rud0 = udiv i32 %pud0, 0
  %rsr0 = srem i32 %psr0, 0
  %rur0 = urem i32 %pur0, 0
  br label %if.end

if.end:                                           ; preds = %if.then, %for.body
  %ysd.0 = phi i32 [ %rsd, %if.then ], [ %psd, %for.body ]
  %yud.0 = phi i32 [ %rud, %if.then ], [ %pud, %for.body ]
  %ysr.0 = phi i32 [ %rsr, %if.then ], [ %psr, %for.body ]
  %yur.0 = phi i32 [ %rur, %if.then ], [ %pur, %for.body ]
  %ysd0.0 = phi i32 [ %rsd0, %if.then ], [ %psd0, %for.body ]
  %yud0.0 = phi i32 [ %rud0, %if.then ], [ %pud0, %for.body ]
  %ysr0.0 = phi i32 [ %rsr0, %if.then ], [ %psr0, %for.body ]
  %yur0.0 = phi i32 [ %rur0, %if.then ], [ %pur0, %for.body ]
  store i32 %ysd.0, ptr %isd, align 4
  store i32 %yud.0, ptr %iud, align 4
  store i32 %ysr.0, ptr %isr, align 4
  store i32 %yur.0, ptr %iur, align 4
  store i32 %ysd0.0, ptr %isd0, align 4
  store i32 %yud0.0, ptr %iud0, align 4
  store i32 %ysr0.0, ptr %isr0, align 4
  store i32 %yur0.0, ptr %iur0, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 128
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}
