; RUN: opt -passes='require<phi-values>,gvn' -S < %s | FileCheck %s

declare noalias i8* @malloc(i64)

; Detecting that %s is fully redundant should let us detect that %w is partially
; redundant.
define void @fn1(i32** noalias %start, i32* %width, i32 %h) {
; CHECK-LABEL: @fn1
entry:
  %call = tail call noalias i8* @malloc(i64 1024)
  %call.cast = bitcast i8* %call to i32*
  store i32* %call.cast, i32** %start, align 8
  br label %preheader

preheader:
  %cmp = icmp slt i32 1, %h
  br i1 %cmp, label %body, label %exit

; CHECK-LABEL: preheader.body_crit_edge:
; CHECK: load i32, i32* %width, align 8

; CHECK-LABEL: body:
; CHECK-NOT: load i32*, i32** %start, align 8
; CHECK-NOT: load i32, i32* %width, align 8
body:
  %j = phi i32 [ 0, %preheader ], [ %j.next, %body ]
  %s = load i32*, i32** %start, align 8
  %idx = getelementptr inbounds i32, i32* %s, i64 0
  store i32 0, i32* %idx, align 4
  %j.next = add nuw nsw i32 %j, 1
  %w = load i32, i32* %width, align 8
  %cmp3 = icmp slt i32 %j.next, %w
  br i1 %cmp3, label %body, label %preheader

exit:
  ret void
}

; %s is fully redundant but has more than one available value. Detecting that
; %w is partially redundant requires alias analysis that can analyze those
; values.
define void @fn2(i32** noalias %start, i32* %width, i32 %h, i32 %arg) {
; CHECK-LABEL: @fn2
entry:
  %call = tail call noalias i8* @malloc(i64 1024)
  %call.cast = bitcast i8* %call to i32*
  %cmp1 = icmp slt i32 %arg, 0
  br i1 %cmp1, label %if, label %else

if:
  store i32* %call.cast, i32** %start, align 8
  br label %preheader

else:
  %gep = getelementptr inbounds i32, i32* %call.cast, i32 %arg
  store i32* %gep, i32** %start, align 8
  br label %preheader

; CHECK-LABEL: preheader:
; CHECK: %s = phi i32* [ %s, %body ], [ %gep, %else ], [ %call.cast, %if ]

preheader:
  %cmp = icmp slt i32 1, %h
  br i1 %cmp, label %body, label %exit

; CHECK-LABEL: preheader.body_crit_edge:
; CHECK: load i32, i32* %width, align 8

; CHECK-LABEL: body:
; CHECK-NOT: load i32*, i32** %start, align 8
; CHECK-NOT: load i32, i32* %width, align 8
body:
  %j = phi i32 [ 0, %preheader ], [ %j.next, %body ]
  %s = load i32*, i32** %start, align 8
  %idx = getelementptr inbounds i32, i32* %s, i64 0
  store i32 0, i32* %idx, align 4
  %j.next = add nuw nsw i32 %j, 1
  %w = load i32, i32* %width, align 8
  %cmp3 = icmp slt i32 %j.next, %w
  br i1 %cmp3, label %body, label %preheader

exit:
  ret void
}
