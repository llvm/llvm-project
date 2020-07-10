; RUN: opt < %s -indvars -S | FileCheck %s

target datalayout = "e-m:e-i64:64-n32:64"

define dso_local void @Widen_i32_i8ptr() local_unnamed_addr {
; CHECK-LABEL: @Widen_i32_i8ptr(
; CHECK: phi i8*
; CHECK: phi i32
entry:
  %ptrids = alloca [15 x i8*], align 8
  %arraydecay2032 = getelementptr inbounds [15 x i8*], [15 x i8*]* %ptrids, i64 0, i64 0
  store i8** %arraydecay2032, i8*** inttoptr (i64 8 to i8***), align 8
  br label %for.cond2106

for.cond2106:                                     ; preds = %for.cond2106, %entry
  %gid.0 = phi i8* [ null, %entry ], [ %incdec.ptr, %for.cond2106 ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc2117, %for.cond2106 ]
  %incdec.ptr = getelementptr inbounds i8, i8* %gid.0, i64 1
  %idxprom2114 = zext i32 %i.0 to i64
  %arrayidx2115 = getelementptr inbounds [15 x i8*], [15 x i8*]* %ptrids, i64 0, i64 %idxprom2114
  store i8* %gid.0, i8** %arrayidx2115, align 8
  %inc2117 = add nuw nsw i32 %i.0, 1
  br label %for.cond2106
}
