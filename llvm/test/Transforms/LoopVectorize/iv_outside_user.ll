; RUN: opt -S -loop-vectorize -force-vector-interleave=1 -force-vector-width=2 < %s | FileCheck %s

; CHECK-LABEL: @postinc
; CHECK-LABEL: scalar.ph:
; CHECK: %bc.resume.val = phi i32 [ %n.vec, %middle.block ], [ 0, %entry ]
; CHECK-LABEL: for.end:
; CHECK: %[[RET:.*]] = phi i32 [ {{.*}}, %for.body ], [ %n.vec, %middle.block ]
; CHECK: ret i32 %[[RET]]
define i32 @postinc(i32 %k)  {
entry:
  br label %for.body

for.body:
  %inc.phi = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %inc = add nsw i32 %inc.phi, 1
  %cmp = icmp eq i32 %inc, %k
  br i1 %cmp, label %for.end, label %for.body

for.end:
  ret i32 %inc
}

; CHECK-LABEL: @preinc
; CHECK-LABEL: middle.block:
; CHECK: %3 = sub i32 %n.vec, 1
; CHECK: %ind.escape = add i32 0, %3
; CHECK-LABEL: scalar.ph:
; CHECK: %bc.resume.val = phi i32 [ %n.vec, %middle.block ], [ 0, %entry ]
; CHECK-LABEL: for.end:
; CHECK: %[[RET:.*]] = phi i32 [ {{.*}}, %for.body ], [ %ind.escape, %middle.block ]
; CHECK: ret i32 %[[RET]]
define i32 @preinc(i32 %k)  {
entry:
  br label %for.body

for.body:
  %inc.phi = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %inc = add nsw i32 %inc.phi, 1
  %cmp = icmp eq i32 %inc, %k
  br i1 %cmp, label %for.end, label %for.body

for.end:
  ret i32 %inc.phi
}

; CHECK-LABEL: @constpre
; CHECK-LABEL: for.end:
; CHECK: %[[RET:.*]] = phi i32 [ {{.*}}, %for.body ], [ 2, %middle.block ]
; CHECK: ret i32 %[[RET]]
define i32 @constpre()  {
entry:
  br label %for.body

for.body:
  %inc.phi = phi i32 [ 32, %entry ], [ %inc, %for.body ]
  %inc = sub nsw i32 %inc.phi, 2
  %cmp = icmp eq i32 %inc, 0
  br i1 %cmp, label %for.end, label %for.body

for.end:
  ret i32 %inc.phi
}

; CHECK-LABEL: @geppre
; CHECK-LABEL: middle.block:
; CHECK: %ind.escape = getelementptr i32, i32* %ptr, i64 124
; CHECK-LABEL: for.end:
; CHECK: %[[RET:.*]] = phi i32* [ {{.*}}, %for.body ], [ %ind.escape, %middle.block ]
; CHECK: ret i32* %[[RET]]
define i32* @geppre(i32* %ptr) {
entry:
  br label %for.body

for.body:
  %inc.phi = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %ptr.phi = phi i32* [ %ptr, %entry ], [ %inc.ptr, %for.body ]
  %inc = add nsw i32 %inc.phi, 1
  %inc.ptr = getelementptr i32, i32* %ptr.phi, i32 4
  %cmp = icmp eq i32 %inc, 32
  br i1 %cmp, label %for.end, label %for.body

for.end:
  ret i32* %ptr.phi
}

; CHECK-LABEL: @both
; CHECK-LABEL: middle.block:
; CHECK: %[[END:.*]] = sub i64 %n.vec, 1
; CHECK: %ind.escape = getelementptr i32, i32* %base, i64 %[[END]]
; CHECK-LABEL: for.end:
; CHECK: %[[RET:.*]] = phi i32* [ %inc.lag1, %for.body ], [ %ind.escape, %middle.block ]
; CHECK: ret i32* %[[RET]]

define i32* @both(i32 %k)  {
entry:
  %base = getelementptr inbounds i32, i32* undef, i64 1
  br label %for.body

for.body:
  %inc.phi = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %inc.lag1 = phi i32* [ %base, %entry ], [ %tmp, %for.body]
  %inc.lag2 = phi i32* [ undef, %entry ], [ %inc.lag1, %for.body]  
  %tmp = getelementptr inbounds i32, i32* %inc.lag1, i64 1    
  %inc = add nsw i32 %inc.phi, 1
  %cmp = icmp eq i32 %inc, %k
  br i1 %cmp, label %for.end, label %for.body

for.end:
  ret i32* %inc.lag1
}
