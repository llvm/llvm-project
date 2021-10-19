; RUN: opt %loadPolly -polly-codegen -S \
; RUN: -polly-invariant-load-hoisting=true < %s | FileCheck %s
;
; This crashed our codegen at some point, verify it runs through
;
; CHECK: polly.start
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

%struct.colocated_params = type { i32, i32, i32, [6 x [33 x i64]], i8***, i64***, i16****, i8**, [6 x [33 x i64]], i8***, i64***, i16****, i8**, [6 x [33 x i64]], i8***, i64***, i16****, i8**, i8, i8** }
%struct.storable_picture9 = type { i32, i32, i32, i32, i32, [50 x [6 x [33 x i64]]], [50 x [6 x [33 x i64]]], [50 x [6 x [33 x i64]]], [50 x [6 x [33 x i64]]], i32, i32, i32, i32, i32, i32, i32, i32, i32, i16, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i16**, i16***, i8*, i16**, i8***, i64***, i64***, i16****, i8**, i8**, %struct.storable_picture9*, %struct.storable_picture9*, %struct.storable_picture9*, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [2 x i32], i32, %struct.DecRefPicMarking_s*, i32 }
%struct.DecRefPicMarking_s = type { i32, i32, i32, i32, i32, %struct.DecRefPicMarking_s* }

; Function Attrs: nounwind uwtable
define void @compute_colocated(%struct.colocated_params* %p) #0 {
entry:
  %tmp = load %struct.storable_picture9*, %struct.storable_picture9** undef, align 8
  br label %for.body.393

for.body.393:                                     ; preds = %if.end.549, %entry
  br i1 undef, label %if.then.397, label %if.else.643

if.then.397:                                      ; preds = %for.body.393
  %ref_idx456 = getelementptr inbounds %struct.storable_picture9, %struct.storable_picture9* %tmp, i64 0, i32 36
  %tmp1 = load i8***, i8**** %ref_idx456, align 8
  %tmp2 = load i8**, i8*** %tmp1, align 8
  %arrayidx458 = getelementptr inbounds i8*, i8** %tmp2, i64 0
  %tmp3 = load i8*, i8** %arrayidx458, align 8
  %arrayidx459 = getelementptr inbounds i8, i8* %tmp3, i64 0
  %tmp4 = load i8, i8* %arrayidx459, align 1
  %cmp461 = icmp eq i8 %tmp4, -1
  br i1 %cmp461, label %if.then.463, label %if.else.476

if.then.463:                                      ; preds = %if.then.397
  br label %if.end.501

if.else.476:                                      ; preds = %if.then.397
  %ref_id491 = getelementptr inbounds %struct.storable_picture9, %struct.storable_picture9* %tmp, i64 0, i32 38
  %tmp5 = load i64***, i64**** %ref_id491, align 8
  br label %if.end.501

if.end.501:                                       ; preds = %if.else.476, %if.then.463
  %tmp6 = load i8***, i8**** %ref_idx456, align 8
  %arrayidx505 = getelementptr inbounds i8**, i8*** %tmp6, i64 1
  %tmp7 = load i8**, i8*** %arrayidx505, align 8
  %arrayidx506 = getelementptr inbounds i8*, i8** %tmp7, i64 0
  %tmp8 = load i8*, i8** %arrayidx506, align 8
  %arrayidx507 = getelementptr inbounds i8, i8* %tmp8, i64 0
  %tmp9 = load i8, i8* %arrayidx507, align 1
  %cmp509 = icmp eq i8 %tmp9, -1
  %ref_idx514 = getelementptr inbounds %struct.colocated_params, %struct.colocated_params* %p, i64 0, i32 4
  %tmp10 = load i8***, i8**** %ref_idx514, align 8
  %arrayidx515 = getelementptr inbounds i8**, i8*** %tmp10, i64 1
  %tmp11 = load i8**, i8*** %arrayidx515, align 8
  %arrayidx516 = getelementptr inbounds i8*, i8** %tmp11, i64 0
  %tmp12 = load i8*, i8** %arrayidx516, align 8
  %arrayidx517 = getelementptr inbounds i8, i8* %tmp12, i64 0
  br i1 %cmp509, label %if.then.511, label %if.else.524

if.then.511:                                      ; preds = %if.end.501
  br label %if.end.549

if.else.524:                                      ; preds = %if.end.501
  store i8 %tmp9, i8* %arrayidx517, align 1
  %ref_id539 = getelementptr inbounds %struct.storable_picture9, %struct.storable_picture9* %tmp, i64 0, i32 38
  %tmp13 = load i64***, i64**** %ref_id539, align 8
  br label %if.end.549

if.end.549:                                       ; preds = %if.else.524, %if.then.511
  br label %for.body.393

if.else.643:                                      ; preds = %for.body.393
  unreachable
}
