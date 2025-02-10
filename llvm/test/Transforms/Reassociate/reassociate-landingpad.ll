; Reassociate used to move the negation of $time_1_P14.0 above the
; landingpad.
;
; RUN: opt -passes=reassociate -disable-output < %s
;
; ModuleID = 'bugpoint-reduced-simplified.bc'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%"lp.2234.4378.7378.12512.15079.17646.20213.22780.25347.27914.40747.53580.74113.76680.86947.89514.92081.94648.115163.130561" = type { ptr, i32 }
%__type_info.2235.4379.7379.12513.15080.17647.20214.22781.25348.27915.40748.53581.74114.76681.86948.89515.92082.94649.115162.130560 = type { ptr, ptr }

declare i32 @__gxx_personality_v0(...)

declare void @b() #0

define void @a(i1 %arg) #0 personality ptr @__gxx_personality_v0 {
", bb1":
  invoke void @b()
          to label %invoke.cont unwind label %"bb22"

", bb8":                                          ; preds = %invoke.cont
  invoke void @c()
          to label %invoke.cont25 unwind label %"bb22"

", bb15":                                         ; preds = %invoke.cont
  ret void

"bb22":     ; preds = %", bb8", %", bb1"
  %"$time_1_P14.0" = phi i64 [ undef, %", bb8" ], [ undef, %", bb1" ]
  %0 = landingpad %"lp.2234.4378.7378.12512.15079.17646.20213.22780.25347.27914.40747.53580.74113.76680.86947.89514.92081.94648.115163.130561"
          cleanup
          catch ptr null
  %r79 = sub i64 0, %"$time_1_P14.0"
  %r81 = add i64 %r79, undef
  %r93 = add i64 %r81, undef
  %r95 = sub i64 %r93, %"$time_1_P14.0"
  %r98 = icmp ult i64 %r95, undef
  unreachable

invoke.cont:                                      ; preds = %", bb1"
  br i1 %arg, label %", bb15", label %", bb8"

invoke.cont25:                                    ; preds = %", bb8"
  unreachable
}

declare void @c() #0

attributes #0 = { "frame-pointer"="non-leaf" }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"Debug Info Version", i32 3}
