; RUN: opt -loop-reduce < %s
; we used to crash on this one

declare i8* @_Znwm()
declare i32 @__gxx_personality_v0(...)
declare void @g()
define void @f() personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
bb0:
  br label %bb1
bb1:
  %v0 = phi i64 [ 0, %bb0 ], [ %v1, %bb1 ]
  %v1 = add nsw i64 %v0, 1
  br i1 undef, label %bb2, label %bb1
bb2:
  %v2 = icmp eq i64 %v0, 0
  br i1 %v2, label %bb6, label %bb3
bb3:
  %v3 = invoke noalias i8* @_Znwm()
          to label %bb5 unwind label %bb4
bb4:
  %v4 = landingpad { i8*, i32 }
          cleanup
  br label %bb9
bb5:
  %v5 = bitcast i8* %v3 to i32**
  %add.ptr.i = getelementptr inbounds i32*, i32** %v5, i64 %v0
  br label %bb6
bb6:
  %v6 = phi i32** [ null, %bb2 ], [ %add.ptr.i, %bb5 ]
  invoke void @g()
          to label %bb7 unwind label %bb8
bb7:
  unreachable
bb8:
  %v7 = landingpad { i8*, i32 }
          cleanup
  br label %bb9
bb9:
  resume { i8*, i32 } zeroinitializer
}


define void @h() personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
bb1:
  invoke void @g() optsize
          to label %bb2 unwind label %bb5
bb2:
  %arrayctor.cur = phi i8* [ undef, %bb1 ], [ %arrayctor.next, %bb3 ]
  invoke void @g() optsize
          to label %bb3 unwind label %bb6
bb3:
  %arrayctor.next = getelementptr inbounds i8, i8* %arrayctor.cur, i64 1
  br label %bb2
bb4:
  ret void
bb5:
  %tmp = landingpad { i8*, i32 }
          cleanup
  invoke void @g() optsize
          to label %bb4 unwind label %bb7
bb6:
  %tmp1 = landingpad { i8*, i32 }
          cleanup
  %arraydestroy.isempty = icmp eq i8* undef, %arrayctor.cur
  ret void
bb7:
  %lpad.nonloopexit = landingpad { i8*, i32 }
          catch i8* null
  ret void
}

; PR17425
define void @i() {
entry:
  br label %while.cond

while.cond:                                       ; preds = %while.cond, %entry
  %c.0 = phi i16* [ undef, %entry ], [ %incdec.ptr, %while.cond ]
  %incdec.ptr = getelementptr inbounds i16, i16* %c.0, i64 1
  br i1 undef, label %while.cond1, label %while.cond

while.cond1:                                      ; preds = %while.cond1, %while.cond
  %c.1 = phi i16* [ %incdec.ptr5, %while.cond1 ], [ %c.0, %while.cond ]
  %incdec.ptr5 = getelementptr inbounds i16, i16* %c.1, i64 1
  br i1 undef, label %while.cond7, label %while.cond1

while.cond7:                                      ; preds = %while.cond7, %while.cond1
  %0 = phi i16* [ %incdec.ptr10, %while.cond7 ], [ %c.1, %while.cond1 ]
  %incdec.ptr10 = getelementptr inbounds i16, i16* %0, i64 1
  br i1 undef, label %while.cond12.preheader, label %while.cond7

while.cond12.preheader:                           ; preds = %while.cond7
  br i1 undef, label %while.end16, label %while.body13.lr.ph

while.body13:                                     ; preds = %if.else, %while.body13.lr.ph
  %1 = phi i16* [ %2, %while.body13.lr.ph ], [ %incdec.ptr15, %if.else ]
  br i1 undef, label %while.cond12.outer.loopexit, label %if.else

while.cond12.outer.loopexit:                      ; preds = %while.body13
  br i1 undef, label %while.end16, label %while.body13.lr.ph

while.body13.lr.ph:                               ; preds = %while.cond12.outer.loopexit, %while.cond12.preheader
  %2 = phi i16* [ %1, %while.cond12.outer.loopexit ], [ undef, %while.cond12.preheader ]
  br label %while.body13

if.else:                                          ; preds = %while.body13
  %incdec.ptr15 = getelementptr inbounds i16, i16* %1, i64 1
  %cmp = icmp eq i16* %incdec.ptr15, %0
  br i1 %cmp, label %while.end16, label %while.body13

while.end16:                                      ; preds = %if.else, %while.cond12.outer.loopexit, %while.cond12.preheader
  ret void
}
