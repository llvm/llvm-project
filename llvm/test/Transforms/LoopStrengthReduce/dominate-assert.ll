; RUN: opt -loop-reduce %s
; we used to crash on this one

declare i8* @_Znwm()
declare i32 @__gxx_personality_v0(...)
declare void @g()
define void @f() {
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
  %v4 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
          cleanup
  br label %bb9
bb5:
  %v5 = bitcast i8* %v3 to i32**
  %add.ptr.i = getelementptr inbounds i32** %v5, i64 %v0
  br label %bb6
bb6:
  %v6 = phi i32** [ null, %bb2 ], [ %add.ptr.i, %bb5 ]
  invoke void @g()
          to label %bb7 unwind label %bb8
bb7:
  unreachable
bb8:
  %v7 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
          cleanup
  br label %bb9
bb9:
  resume { i8*, i32 } zeroinitializer
}


define void @h() {
bb1:
  invoke void @g() optsize
          to label %bb2 unwind label %bb5
bb2:
  %arrayctor.cur = phi i8* [ undef, %bb1 ], [ %arrayctor.next, %bb3 ]
  invoke void @g() optsize
          to label %bb3 unwind label %bb6
bb3:
  %arrayctor.next = getelementptr inbounds i8* %arrayctor.cur, i64 1
  br label %bb2
bb4:
  ret void
bb5:
  %tmp = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
          cleanup
  invoke void @g() optsize
          to label %bb4 unwind label %bb7
bb6:
  %tmp1 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
          cleanup
  %arraydestroy.isempty = icmp eq i8* undef, %arrayctor.cur
  ret void
bb7:
  %lpad.nonloopexit = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
          catch i8* null
  ret void
}
