; RUN: opt -S -passes=loop-unroll -unroll-force-peel-count=1 -verify-dom-info <%s

; Check if loop composed of several BBs is peeled correctly.

declare void @funcb()
@Comma = external global i8
define void @funca(ptr readnone %b, ptr readnone %e) {
entry:
  %cmp2 = icmp eq ptr %b, %e
  br i1 %cmp2, label %for.end, label %for.body.preheader

for.body.preheader:
  br label %for.body

for.body:
  %b.addr.03 = phi ptr [ %incdec.ptr, %for.inc ], [ %b, %for.body.preheader ]
  %0 = load i8, ptr @Comma
  %tobool = icmp eq i8 %0, 0
  br i1 %tobool, label %for.inc, label %if.then

if.then:
  tail call void @funcb()
  store i8 1, ptr @Comma
  br label %for.inc

for.inc:
  %incdec.ptr = getelementptr inbounds i8, ptr %b.addr.03, i64 1
  %cmp = icmp eq ptr %incdec.ptr, %e
  br i1 %cmp, label %for.end, label %for.body

for.end:
  ret void
}

; CHECK-LABEL: @funca

; Peeled iteration
; CHECK: %[[REG1:[0-9]+]] = load i8, ptr @Comma
; CHECK: %[[REG2:.*]] = icmp eq i8 %[[REG1]], 0
; CHECK: br i1 %[[REG2]], label %{{.*}}, label %[[IFTHEN:.*]]
; CHECK: [[IFTHEN]]:
; CHECK: call void @funcb()
; CHECK: store i8 1, ptr @Comma
; CHECK: br label %[[FORINC]]
; CHECK: [[FORINC]]:
; CHECK: %[[REG3:.*]] = getelementptr inbounds i8, ptr %b, i64 1
; CHECK: %[[REG4:.*]] = icmp eq ptr %[[REG3]], %e
; CHECK: br i1 %[[REG4]]

; main body
; CHECK: %[[REG1b:.*]] = load i8, ptr @Comma
; CHECK: %[[REG2b:.*]] = icmp eq i8 %[[REG1b]], 0
; CHECK: br i1 %[[REG2b]], label %{{.*}}, label %[[IFTHENb:.*]]
; CHECK: [[IFTHENb]]:
; CHECK: call void @funcb()
; CHECK: store i8 1, ptr @Comma
; CHECK: br label %[[FORINCb]]
; CHECK: [[FORINCb]]:
; CHECK: %[[REG3b:.*]] = getelementptr inbounds i8, ptr %b, i64 1
; CHECK: %[[REG4b:.*]] = icmp eq ptr %[[REG3b]], %e
; CHECK: br i1 %[[REG4b]]
