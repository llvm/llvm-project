; RUN: llc < %s
; PR10733
declare void @_Znam()

define void @_ZNK14gIndexOdometer15AfterExcisionOfERi() uwtable align 2 {
_ZN6Gambit5ArrayIiEC2Ej.exit36:
  br label %"9"

"9":                                              ; preds = %"10", %_ZN6Gambit5ArrayIiEC2Ej.exit36
  %indvar82 = phi i64 [ 0, %_ZN6Gambit5ArrayIiEC2Ej.exit36 ], [ %tmp85, %"10" ]
  %tmp85 = add i64 %indvar82, 1
  %tmp = trunc i64 %tmp85 to i32
  invoke void @_ZNK14gIndexOdometer9NoIndicesEv()
          to label %"10" unwind label %lpad27

"10":                                             ; preds = %"9"
  invoke void @_Znam()
          to label %"9" unwind label %lpad27

lpad27:                                           ; preds = %"10", %"9"
  %0 = phi i32 [ undef, %"9" ], [ %tmp, %"10" ]
  %1 = landingpad { i8*, i32 } personality i32 (i32, i64, i8*, i8*)* @__gxx_personality_v0
          cleanup
  resume { i8*, i32 } zeroinitializer
}

declare void @_ZNK14gIndexOdometer9NoIndicesEv()

declare i32 @__gxx_personality_v0(i32, i64, i8*, i8*)
