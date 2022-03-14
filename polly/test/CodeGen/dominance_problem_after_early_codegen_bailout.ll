; RUN: opt %loadPolly -polly-codegen -disable-output < %s
;
; This caused dominance problems at some point as we do bail out during
; code generation. Just verify it runs through.
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

%struct.hashheader.0.5.10.165.180.185 = type { i16, i16, i16, i16, i16, i16, i32, i32, i32, i32, i32, i32, i32, i32, i32, [5 x i8], [13 x i8], i8, i8, i8, [228 x i16], [228 x i8], [228 x i8], [228 x i8], [228 x i8], [228 x i8], [228 x i8], [128 x i8], [100 x [11 x i8]], [100 x i32], [100 x i32], i16 }

@hashheader = external global %struct.hashheader.0.5.10.165.180.185, align 4

; Function Attrs: nounwind uwtable
define void @strtoichar(i8* %in) #0 {
entry:
  br i1 undef, label %land.rhs, label %for.end

land.rhs:                                         ; preds = %for.inc, %entry
  %in.addr.012 = phi i8* [ undef, %for.inc ], [ %in, %entry ]
  %0 = load i8, i8* %in.addr.012, align 1
  br i1 undef, label %for.end, label %for.body

for.body:                                         ; preds = %land.rhs
  %idxprom = zext i8 %0 to i64
  %arrayidx = getelementptr inbounds %struct.hashheader.0.5.10.165.180.185, %struct.hashheader.0.5.10.165.180.185* @hashheader, i64 0, i32 27, i64 %idxprom
  %1 = load i8, i8* %arrayidx, align 1
  %tobool = icmp eq i8 %1, 0
  br i1 %tobool, label %if.else, label %land.rhs.7

land.rhs.7:                                       ; preds = %for.body
  tail call void @stringcharlen()
  br i1 undef, label %if.then, label %if.else

if.then:                                          ; preds = %land.rhs.7
  br label %for.inc

if.else:                                          ; preds = %land.rhs.7, %for.body
  %2 = load i8, i8* %in.addr.012, align 1
  br label %for.inc

for.inc:                                          ; preds = %if.else, %if.then
  %len.1 = phi i32 [ 0, %if.else ], [ undef, %if.then ]
  br i1 undef, label %land.rhs, label %for.end

for.end:                                          ; preds = %for.inc, %land.rhs, %entry
  ret void
}

; Function Attrs: nounwind uwtable
declare void @stringcharlen() #0
