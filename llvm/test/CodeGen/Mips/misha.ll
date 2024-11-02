; RUN: llc  -march=mipsel -mattr=mips16 -relocation-model=pic -O3 < %s | FileCheck %s -check-prefix=16

define i32 @sumc(ptr nocapture %to, ptr nocapture %from, i32) nounwind {
entry:
  %sext = shl i32 %0, 16
  %conv = ashr exact i32 %sext, 16
  %cmp8 = icmp eq i32 %conv, 0
  br i1 %cmp8, label %for.end, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %entry
  %.pre = load i8, ptr %to, align 1
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %1 = phi i8 [ %.pre, %for.body.lr.ph ], [ %conv4, %for.body ]
  %i.010 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.body ]
  %from.addr.09 = phi ptr [ %from, %for.body.lr.ph ], [ %incdec.ptr, %for.body ]
  %incdec.ptr = getelementptr inbounds i8, ptr %from.addr.09, i32 1
  %2 = load i8, ptr %from.addr.09, align 1
  %conv27 = zext i8 %2 to i32
  %conv36 = zext i8 %1 to i32
  %add = add nsw i32 %conv36, %conv27
  %conv4 = trunc i32 %add to i8
  store i8 %conv4, ptr %to, align 1
  %inc = add nsw i32 %i.010, 1
  %cmp = icmp eq i32 %inc, %conv
  br i1 %cmp, label %for.end, label %for.body
; 16-LABEL: sumc:
; 16: 	lbu	${{[0-9]+}}, 0(${{[0-9]+}})
; 16: 	lbu	${{[0-9]+}}, 0(${{[0-9]+}})
; 16-LABEL: sum:
; 16: 	lhu	${{[0-9]+}}, 0(${{[0-9]+}})
; 16: 	lhu	${{[0-9]+}}, 0(${{[0-9]+}})

for.end:                                          ; preds = %for.body, %entry
  ret i32 undef
}

define i32 @sum(ptr nocapture %to, ptr nocapture %from, i32) nounwind {
entry:
  %sext = shl i32 %0, 16
  %conv = ashr exact i32 %sext, 16
  %cmp8 = icmp eq i32 %conv, 0
  br i1 %cmp8, label %for.end, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %entry
  %.pre = load i16, ptr %to, align 2
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %1 = phi i16 [ %.pre, %for.body.lr.ph ], [ %conv4, %for.body ]
  %i.010 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.body ]
  %from.addr.09 = phi ptr [ %from, %for.body.lr.ph ], [ %incdec.ptr, %for.body ]
  %incdec.ptr = getelementptr inbounds i16, ptr %from.addr.09, i32 1
  %2 = load i16, ptr %from.addr.09, align 2
  %conv27 = zext i16 %2 to i32
  %conv36 = zext i16 %1 to i32
  %add = add nsw i32 %conv36, %conv27
  %conv4 = trunc i32 %add to i16
  store i16 %conv4, ptr %to, align 2
  %inc = add nsw i32 %i.010, 1
  %cmp = icmp eq i32 %inc, %conv
  br i1 %cmp, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret i32 undef
}


