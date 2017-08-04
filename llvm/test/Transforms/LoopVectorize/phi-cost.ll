; REQUIRES: asserts
; RUN: opt < %s -loop-vectorize -force-vector-width=2 -force-vector-interleave=1 -instcombine -debug-only=loop-vectorize -disable-output -print-after=instcombine 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"

; CHECK-LABEL: phi_two_incoming_values
; CHECK:       LV: Found an estimated cost of 1 for VF 2 For instruction: %i = phi i64 [ %i.next, %if.end ], [ 0, %entry ]
; CHECK:       LV: Found an estimated cost of 1 for VF 2 For instruction: %tmp5 = phi i32 [ %tmp1, %for.body ], [ %tmp4, %if.then ]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], %vector.body ]
; CHECK:         [[WIDE_LOAD:%.*]] = load <2 x i32>, <2 x i32>* {{.*}}
; CHECK:         [[TMP5:%.*]] = icmp sgt <2 x i32> [[WIDE_LOAD]], zeroinitializer
; CHECK-NEXT:    [[TMP6:%.*]] = add <2 x i32> [[WIDE_LOAD]], <i32 1, i32 1>
; CHECK-NEXT:    [[PREDPHI:%.*]] = select <2 x i1> [[TMP5]], <2 x i32> [[TMP6]], <2 x i32> [[WIDE_LOAD]]
; CHECK:         store <2 x i32> [[PREDPHI]], <2 x i32>* {{.*}}
; CHECK-NEXT:    [[INDEX_NEXT]] = add i64 [[INDEX]], 2
;
define void @phi_two_incoming_values(i32* %a, i32* %b, i64 %n) {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ %i.next, %if.end ], [ 0, %entry ]
  %tmp0 = getelementptr inbounds i32, i32* %a, i64 %i
  %tmp1 = load i32, i32* %tmp0, align 4
  %tmp2 = getelementptr inbounds i32, i32* %b, i64 %i
  %tmp3 = icmp sgt i32 %tmp1, 0
  br i1 %tmp3, label %if.then, label %if.end

if.then:
  %tmp4 = add i32 %tmp1, 1
  br label %if.end

if.end:
  %tmp5 = phi i32 [ %tmp1, %for.body ], [ %tmp4, %if.then ]
  store i32 %tmp5, i32* %tmp2, align 4
  %i.next = add i64 %i, 1
  %cond = icmp eq i64 %i, %n
  br i1 %cond, label %for.end, label %for.body

for.end:
  ret void
}

; CHECK-LABEL: phi_three_incoming_values
; CHECK:       LV: Found an estimated cost of 1 for VF 2 For instruction: %i = phi i64 [ %i.next, %if.end ], [ 0, %entry ]
; CHECK:       LV: Found an estimated cost of 2 for VF 2 For instruction: %tmp8 = phi i32 [ 9, %for.body ], [ 3, %if.then ], [ %tmp7, %if.else ]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], %vector.body ]
; CHECK:         [[PREDPHI:%.*]] = select <2 x i1> {{.*}}, <2 x i32> <i32 3, i32 3>, <2 x i32> <i32 9, i32 9>
; CHECK:         [[PREDPHI7:%.*]] = select <2 x i1> {{.*}}, <2 x i32> {{.*}}, <2 x i32> [[PREDPHI]]
; CHECK:         store <2 x i32> [[PREDPHI7]], <2 x i32>* {{.*}}
; CHECK-NEXT:    [[INDEX_NEXT]] = add i64 [[INDEX]], 2
;
define void @phi_three_incoming_values(i32* %a, i32* %b, i64 %n) {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ %i.next, %if.end ], [ 0, %entry ]
  %tmp0 = getelementptr inbounds i32, i32* %a, i64 %i
  %tmp1 = load i32, i32* %tmp0, align 4
  %tmp2 = getelementptr inbounds i32, i32* %b, i64 %i
  %tmp3 = load i32, i32* %tmp2, align 4
  %tmp4 = icmp sgt i32 %tmp1, %tmp3
  br i1 %tmp4, label %if.then, label %if.end

if.then:
  %tmp5 = icmp sgt i32 %tmp1, 19
  br i1 %tmp5, label %if.end, label %if.else

if.else:
  %tmp6 = icmp slt i32 %tmp3, 4
  %tmp7 = select i1 %tmp6, i32 4, i32 5
  br label %if.end

if.end:
  %tmp8 = phi i32 [ 9, %for.body ], [ 3, %if.then ], [ %tmp7, %if.else ]
  store i32 %tmp8, i32* %tmp0, align 4
  %i.next = add i64 %i, 1
  %cond = icmp eq i64 %i, %n
  br i1 %cond, label %for.end, label %for.body

for.end:
  ret void
}
