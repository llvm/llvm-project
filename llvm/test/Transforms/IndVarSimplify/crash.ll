; RUN: opt -passes=indvars -disable-output < %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

declare i32 @putchar(i8) nounwind

define void @t2(ptr %P) nounwind {
; <label>:0
  br label %1

; <label>:1                                       ; preds = %1, %0
  %2 = phi double [ 9.000000e+00, %0 ], [ %4, %1 ] ; <double> [#uses=1]
  %3 = tail call i32 @putchar(i8 72)              ; <i32> [#uses=0]
  %4 = fadd double %2, -1.000000e+00              ; <double> [#uses=2]
  %5 = fcmp ult double %4, 0.000000e+00           ; <i1> [#uses=1]
  store i1 %5, ptr %P
  br i1 %5, label %6, label %1

; <label>:6                                       ; preds = %1
  ret void
}

; PR7562
define void @fannkuch(i1 %arg) nounwind {
entry:                                              ; preds = %entry
  br label %bb12

bb12:                                             ; preds = %bb29, %entry
  %i.1 = phi i32 [ undef, %entry ], [ %i.0, %bb29 ] ; <i32> [#uses=2]
  %r.1 = phi i32 [ undef, %entry ], [ %r.0, %bb29 ] ; <i32> [#uses=2]
  br i1 %arg, label %bb13, label %bb24

bb13:                                             ; preds = %bb12
  br label %bb24

bb24:                                             ; preds = %bb30, %bb13, %bb12
  %i.2 = phi i32 [ %i.1, %bb13 ], [ %i.0, %bb30 ], [ %i.1, %bb12 ] ; <i32> [#uses=1]
  %r.0 = phi i32 [ %r.1, %bb13 ], [ %2, %bb30 ], [ %r.1, %bb12 ] ; <i32> [#uses=3]
  br label %bb28

bb27:                                             ; preds = %bb28
  %0 = add nsw i32 %i.0, 1                        ; <i32> [#uses=1]
  br label %bb28

bb28:                                             ; preds = %bb27, %bb26
  %i.0 = phi i32 [ %i.2, %bb24 ], [ %0, %bb27 ]   ; <i32> [#uses=4]
  %1 = icmp slt i32 %i.0, %r.0                    ; <i1> [#uses=1]
  br i1 %1, label %bb27, label %bb29

bb29:                                             ; preds = %bb28
  br i1 %arg, label %bb12, label %bb30

bb30:                                             ; preds = %bb29
  %2 = add nsw i32 %r.0, 1                        ; <i32> [#uses=1]
  br label %bb24
}

; PR10770

declare void @__go_panic() noreturn

declare void @__go_undefer()

declare i32 @__gccgo_personality_v0(i32, i64, ptr, ptr)

define void @main.main() uwtable personality ptr @__gccgo_personality_v0 {
entry:
  invoke void @__go_panic() noreturn
          to label %0 unwind label %"5.i"

; <label>:0                                       ; preds = %entry
  unreachable

"3.i":                                            ; preds = %"7.i", %"5.i"
  invoke void @__go_undefer()
          to label %main.f.exit unwind label %"7.i"

"5.i":                                            ; preds = %entry
  %1 = landingpad { ptr, i32 }
          catch ptr null
  br label %"3.i"

"7.i":                                            ; preds = %"3.i"
  %2 = landingpad { ptr, i32 }
          catch ptr null
  br label %"3.i"

main.f.exit:                                      ; preds = %"3.i"
  unreachable
}


; PR13967

define void @f() nounwind ssp {
bb:
  br label %bb4

bb4:
  %tmp = phi i64 [ %tmp5, %bb7 ], [ undef, %bb ]
  %tmp5 = add nsw i64 %tmp, 1
  %extract.t1 = trunc i64 %tmp5 to i32
  br i1 false, label %bb6, label %bb7

bb6:
  br label %bb7

bb7:
  %.off0 = phi i32 [ undef, %bb6 ], [ %extract.t1, %bb4 ]
  %tmp8 = icmp eq i32 %.off0, 0
  br i1 %tmp8, label %bb9, label %bb4

bb9:
  ret void
}

; PR12536
define void @fn1(i1 %arg) noreturn nounwind {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.end, %entry
  %b.0 = phi i32 [ undef, %entry ], [ %conv, %for.end ]
  br label %for.cond1

for.cond1:                                        ; preds = %for.cond1, %for.cond
  %c.0 = phi i32 [ %b.0, %for.cond1 ], [ 0, %for.cond ]
  br i1 %arg, label %for.cond1, label %for.end

for.end:                                          ; preds = %for.cond1
  %cmp2 = icmp slt i32 %c.0, 1
  %conv = zext i1 %cmp2 to i32
  br label %for.cond
}
