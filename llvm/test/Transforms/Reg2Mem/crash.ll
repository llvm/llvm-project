; RUN: opt -passes=reg2mem -disable-output < %s
; PR14782

declare void @f1()

declare i32 @__gxx_personality_sj0(...)

declare void @f2()

declare void @f3()

declare void @f4_()

declare void @_Z12xxxdtsP10xxxpq()

define hidden void @_ZN12xxxyzIi9xxxwLi29ELi0EE4f3NewES0_i() ssp align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_sj0 to i8*) {
bb:
  invoke void @f4_()
          to label %bb1 unwind label %.thread

.thread:                                          ; preds = %bb
  %tmp = landingpad { i8*, i32 }
          cleanup
  br label %bb13

bb1:                                              ; preds = %bb
  invoke void @f1()
          to label %.noexc unwind label %bb10

.noexc:                                           ; preds = %bb1
  invoke void @f4_()
          to label %bb6 unwind label %bb2

bb2:                                              ; preds = %.noexc
  %tmp3 = landingpad { i8*, i32 }
          cleanup
  invoke void @f3()
          to label %.body unwind label %bb4

bb4:                                              ; preds = %bb2
  %tmp5 = landingpad { i8*, i32 }
          catch i8* null
  unreachable

bb6:                                              ; preds = %.noexc
  invoke void @_Z12xxxdtsP10xxxpq()
          to label %_ZN6xxxdIN12xxxyzIi9xxxwLi29ELi0EE4fr1jS3_.exit unwind label %bb10

_ZN6xxxdIN12xxxyzIi9xxxwLi29ELi0EE4fr1jS3_.exit:  ; preds = %bb6
  invoke void @f2()
          to label %bb7 unwind label %bb8

bb7:                                              ; preds = %_ZN6xxxdIN12xxxyzIi9xxxwLi29ELi0EE4fr1jS3_.exit
  ret void

bb8:                                              ; preds = %_ZN6xxxdIN12xxxyzIi9xxxwLi29ELi0EE4fr1jS3_.exit
  %tmp9 = landingpad { i8*, i32 }
          cleanup
  br label %_ZN10xxxpqdlev.exit

bb10:                                             ; preds = %bb6, %bb1
  %.1 = phi i1 [ true, %bb1 ], [ false, %bb6 ]
  %tmp11 = landingpad { i8*, i32 }
          cleanup
  br label %.body

.body:                                            ; preds = %bb10, %bb2
  %.1.lpad-body = phi i1 [ %.1, %bb10 ], [ true, %bb2 ]
  invoke void @f2()
          to label %bb12 unwind label %bb14

bb12:                                             ; preds = %.body
  br i1 %.1.lpad-body, label %bb13, label %_ZN10xxxpqdlev.exit

bb13:                                             ; preds = %bb12, %.thread
  invoke void @xxx_MemFree()
          to label %_ZN10xxxpqdlev.exit unwind label %bb14

_ZN10xxxpqdlev.exit:                              ; preds = %bb13, %bb12, %bb8
  resume { i8*, i32 } undef

bb14:                                             ; preds = %bb13, %.body
  %tmp15 = landingpad { i8*, i32 }
          catch i8* null
  unreachable
}

declare void @xxx_MemFree()
