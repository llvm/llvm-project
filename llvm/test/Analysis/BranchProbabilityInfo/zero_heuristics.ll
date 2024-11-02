; RUN: opt < %s -passes='print<branch-prob>' -disable-output 2>&1 | FileCheck %s

@A = global i32 0, align 4
@B = global i32 0, align 4

; CHECK-LABEL: eq_opaque_minus_one
define void @eq_opaque_minus_one(ptr %base) {
entry:
  %const = bitcast i32 -1 to i32
  %tmp1 = load i32, ptr @B, align 4
  br label %for.body

; CHECK: edge for.body -> if.then probability is 0x30000000 / 0x80000000 = 37.50%
; CHECK: edge for.body -> for.inc probability is 0x50000000 / 0x80000000 = 62.50%
for.body:
  %tmp4 = phi i32 [ %tmp1, %entry ], [ %tmp7, %for.inc ]
  %inc.iv = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %storemerge176.in = getelementptr inbounds i32, ptr %base, i32 %inc.iv
  %storemerge176 = load i32, ptr %storemerge176.in, align 4
  store i32 %storemerge176, ptr @A, align 4
  %cmp20 = icmp eq i32 %storemerge176, %const
  br i1 %cmp20, label %if.then, label %for.inc

if.then:
  %lnot.ext = zext i1 %cmp20 to i32
  store i32 %lnot.ext, ptr @B, align 4
  br label %for.inc

for.inc:
  %tmp7 = phi i32 [ %tmp4, %for.body ], [ %lnot.ext, %if.then ]
  %inc = add nuw nsw i32 %inc.iv, 1
  %cmp9 = icmp ult i32 %inc, 401
  br i1 %cmp9, label %for.body, label %exit

exit:
  ret void
}

; CHECK-LABEL: ne_opaque_minus_one
define void @ne_opaque_minus_one(ptr %base) {
entry:
  %const = bitcast i32 -1 to i32
  %tmp1 = load i32, ptr @B, align 4
  br label %for.body

; CHECK: edge for.body -> if.then probability is 0x50000000 / 0x80000000 = 62.50%
; CHECK: edge for.body -> for.inc probability is 0x30000000 / 0x80000000 = 37.50%
for.body:
  %tmp4 = phi i32 [ %tmp1, %entry ], [ %tmp7, %for.inc ]
  %inc.iv = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %storemerge176.in = getelementptr inbounds i32, ptr %base, i32 %inc.iv
  %storemerge176 = load i32, ptr %storemerge176.in, align 4
  store i32 %storemerge176, ptr @A, align 4
  %cmp20 = icmp ne i32 %storemerge176, %const
  br i1 %cmp20, label %if.then, label %for.inc

if.then:
  %lnot.ext = zext i1 %cmp20 to i32
  store i32 %lnot.ext, ptr @B, align 4
  br label %for.inc

for.inc:
  %tmp7 = phi i32 [ %tmp4, %for.body ], [ %lnot.ext, %if.then ]
  %inc = add nuw nsw i32 %inc.iv, 1
  %cmp9 = icmp ult i32 %inc, 401
  br i1 %cmp9, label %for.body, label %exit

exit:
  ret void
}

; CHECK-LABEL: sgt_opaque_minus_one
define void @sgt_opaque_minus_one(ptr %base) {
entry:
  %const = bitcast i32 -1 to i32
  %tmp1 = load i32, ptr @B, align 4
  br label %for.body

; CHECK: edge for.body -> if.then probability is 0x50000000 / 0x80000000 = 62.50%
; CHECK: edge for.body -> for.inc probability is 0x30000000 / 0x80000000 = 37.50%
for.body:
  %tmp4 = phi i32 [ %tmp1, %entry ], [ %tmp7, %for.inc ]
  %inc.iv = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %storemerge176.in = getelementptr inbounds i32, ptr %base, i32 %inc.iv
  %storemerge176 = load i32, ptr %storemerge176.in, align 4
  store i32 %storemerge176, ptr @A, align 4
  %cmp20 = icmp sgt i32 %storemerge176, %const
  br i1 %cmp20, label %if.then, label %for.inc

if.then:
  %lnot.ext = zext i1 %cmp20 to i32
  store i32 %lnot.ext, ptr @B, align 4
  br label %for.inc

for.inc:
  %tmp7 = phi i32 [ %tmp4, %for.body ], [ %lnot.ext, %if.then ]
  %inc = add nuw nsw i32 %inc.iv, 1
  %cmp9 = icmp ult i32 %inc, 401
  br i1 %cmp9, label %for.body, label %exit

exit:
  ret void
}
