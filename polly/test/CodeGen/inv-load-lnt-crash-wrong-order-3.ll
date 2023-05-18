; RUN: opt %loadPolly -polly-codegen -S \
; RUN: -polly-invariant-load-hoisting=true < %s | FileCheck %s
;
; This crashed our codegen at some point, verify it runs through
;
; CHECK: polly.start
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

%struct.colocated_params = type { i32, i32, i32, [6 x [33 x i64]], ptr, ptr, ptr, ptr, [6 x [33 x i64]], ptr, ptr, ptr, ptr, [6 x [33 x i64]], ptr, ptr, ptr, ptr, i8, ptr }
%struct.storable_picture9 = type { i32, i32, i32, i32, i32, [50 x [6 x [33 x i64]]], [50 x [6 x [33 x i64]]], [50 x [6 x [33 x i64]]], [50 x [6 x [33 x i64]]], i32, i32, i32, i32, i32, i32, i32, i32, i32, i16, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [2 x i32], i32, ptr, i32 }
%struct.DecRefPicMarking_s = type { i32, i32, i32, i32, i32, ptr }

; Function Attrs: nounwind uwtable
define void @compute_colocated(ptr %p) #0 {
entry:
  %tmp = load ptr, ptr undef, align 8
  br label %for.body.393

for.body.393:                                     ; preds = %if.end.549, %entry
  br i1 undef, label %if.then.397, label %if.else.643

if.then.397:                                      ; preds = %for.body.393
  %ref_idx456 = getelementptr inbounds %struct.storable_picture9, ptr %tmp, i64 0, i32 36
  %tmp1 = load ptr, ptr %ref_idx456, align 8
  %tmp2 = load ptr, ptr %tmp1, align 8
  %tmp3 = load ptr, ptr %tmp2, align 8
  %tmp4 = load i8, ptr %tmp3, align 1
  %cmp461 = icmp eq i8 %tmp4, -1
  br i1 %cmp461, label %if.then.463, label %if.else.476

if.then.463:                                      ; preds = %if.then.397
  br label %if.end.501

if.else.476:                                      ; preds = %if.then.397
  %ref_id491 = getelementptr inbounds %struct.storable_picture9, ptr %tmp, i64 0, i32 38
  %tmp5 = load ptr, ptr %ref_id491, align 8
  br label %if.end.501

if.end.501:                                       ; preds = %if.else.476, %if.then.463
  %tmp6 = load ptr, ptr %ref_idx456, align 8
  %arrayidx505 = getelementptr inbounds ptr, ptr %tmp6, i64 1
  %tmp7 = load ptr, ptr %arrayidx505, align 8
  %tmp8 = load ptr, ptr %tmp7, align 8
  %tmp9 = load i8, ptr %tmp8, align 1
  %cmp509 = icmp eq i8 %tmp9, -1
  %ref_idx514 = getelementptr inbounds %struct.colocated_params, ptr %p, i64 0, i32 4
  %tmp10 = load ptr, ptr %ref_idx514, align 8
  %arrayidx515 = getelementptr inbounds ptr, ptr %tmp10, i64 1
  %tmp11 = load ptr, ptr %arrayidx515, align 8
  %tmp12 = load ptr, ptr %tmp11, align 8
  br i1 %cmp509, label %if.then.511, label %if.else.524

if.then.511:                                      ; preds = %if.end.501
  br label %if.end.549

if.else.524:                                      ; preds = %if.end.501
  store i8 %tmp9, ptr %tmp12, align 1
  %ref_id539 = getelementptr inbounds %struct.storable_picture9, ptr %tmp, i64 0, i32 38
  %tmp13 = load ptr, ptr %ref_id539, align 8
  br label %if.end.549

if.end.549:                                       ; preds = %if.else.524, %if.then.511
  br label %for.body.393

if.else.643:                                      ; preds = %for.body.393
  unreachable
}
