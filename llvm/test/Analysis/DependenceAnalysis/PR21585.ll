; RUN: opt < %s -disable-output "-passes=print<da>"                            \
; RUN: "-aa-pipeline=basic-aa,globals-aa" 2>&1 | FileCheck %s
define void @i32_subscript(ptr %a) {
entry:
  br label %for.body

for.body:
  %i = phi i32 [ 0, %entry ], [ %i.inc, %for.body ]
  %a.addr = getelementptr i32, ptr %a, i32 %i
  %a.addr.2 = getelementptr i32, ptr %a, i32 5
  %0 = load i32, ptr %a.addr, align 4
  %1 = add i32 %0, 1
  store i32 %1, ptr %a.addr.2, align 4
  %i.inc = add nsw i32 %i, 1
  %i.inc.ext = sext i32 %i to i64
  %exitcond = icmp ne i64 %i.inc.ext, 100
  br i1 %exitcond, label %for.body, label %for.end

for.end:
  ret void
}
; CHECK: none
; CHECK: anti
; CHECK: output


; Test for a bug, which caused an assert in ScalarEvolution because
; the Dependence Analyzer attempted to zero extend a type to a smaller
; type.

; void t(unsigned int *a, unsigned int n) {
;   for (unsigned int i = 0; i != n; i++) {
;     a[(unsigned short)i] = g;
;  }}

@g = common global i32 0, align 4

define void @t(ptr noalias %a, i32 %n) nounwind {
entry:
  %cmp1 = icmp eq i32 %n, 0
  br i1 %cmp1, label %for.end, label %for.body

for.body:
  %i.02 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %0 = load i32, ptr @g, align 4
  %idxprom = and i32 %i.02, 65535
  %arrayidx = getelementptr inbounds i32, ptr %a, i32 %idxprom
  store i32 %0, ptr %arrayidx, align 4
  %inc = add i32 %i.02, 1
  %cmp = icmp eq i32 %inc, %n
  br i1 %cmp, label %for.end, label %for.body

for.end:
  ret void
}
; CHECK: input
; CHECK: none
; CHECK: output

define void @i16_wrap(ptr %a) {
entry:
  br label %for.body
for.body:
  %i = phi i64 [0, %entry], [%i.inc, %for.inc]
  %i.tr = trunc i64 %i to i16
  %idx = getelementptr i64, ptr %a, i16 %i.tr
  %0 = load i64, ptr %idx
  %1 = add i64 %0, 1
store i64 %1, ptr %idx
  br label %for.inc

for.inc:
  %i.inc = add nuw i64 %i, 1
  %cmp = icmp ult i64 %i.inc, 17179869184
  br i1 %cmp, label %for.body, label %for.end
for.end:
  ret void
}
; CHECK: input
; CHECK: anti
; CHECK: output

define void @i8_stride_wrap(ptr noalias %a, ptr noalias %b) {
entry:
  br label %for.body
for.body:
  %i = phi i32 [1,%entry], [%i.inc, %for.inc]
  %i.tr = trunc i32 %i to i8
  %idx = getelementptr i32, ptr %a, i8 %i.tr
  %idx.2 = getelementptr i32, ptr %b, i32 %i
  %0 = load i32, ptr %idx, align 4
  %1 = add i32 %0, 1
  store i32 %1, ptr %idx.2, align 4
  br label %for.inc

for.inc:
  %i.inc = add nsw i32 %i, 256
  %exitcond = icmp ult i32 %i, 65536
  br i1 %exitcond, label %for.body, label %for.end

for.end:
  ret void
}
; CHECK: input
; CHECK: none
; CHECK: none
