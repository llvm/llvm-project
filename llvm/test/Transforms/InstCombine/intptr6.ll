; RUN: opt < %s  -instcombine -S 
; no crash

%A = type { %B }
%B = type { %C *}
%C = type <{ i32 (...)**, i32, [4 x i8] }>

$foo = comdat any

@bar= external thread_local global %A, align 8

declare i32 @__gxx_personality_v0(...)

; Function Attrs: inlinehint sanitize_memory uwtable
define void @foo() local_unnamed_addr #0 comdat align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %0 = load %C*, %C** getelementptr inbounds (%A, %A* @bar, i64 0, i32 0, i32 0), align 8
  %1 = ptrtoint %C* %0 to i64
  %count.i.i.i23 = getelementptr inbounds %C, %C* %0, i64 0, i32 1
  store i32 0, i32* %count.i.i.i23, align 8
  %2 = invoke i8* @_Znwm() #3
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %entry
  %call.i25 = invoke i8* @_Znwm() #3
          to label %call.i.noexc unwind label %lpad4

call.i.noexc:                                     ; preds = %invoke.cont
  invoke void @lazy()
          to label %invoke.cont5 unwind label %lpad.i

lpad.i:                                           ; preds = %call.i.noexc
  %3 = landingpad { i8*, i32 }
          cleanup
  br label %ehcleanup

invoke.cont5:                                     ; preds = %call.i.noexc
  %4 = ptrtoint i8* %call.i25 to i64
  invoke void @scale()
          to label %invoke.cont16 unwind label %lpad15

invoke.cont16:                                    ; preds = %invoke.cont5
  ret void

lpad:                                             ; preds = %entry
  %5 = landingpad { i8*, i32 }
          cleanup
  unreachable

lpad4:                                            ; preds = %invoke.cont
  %6 = landingpad { i8*, i32 }
          cleanup
  unreachable

ehcleanup:                                        ; preds = %lpad.i
  br label %ehcleanup21

lpad15:                                           ; preds = %invoke.cont5
  %7 = landingpad { i8*, i32 }
          cleanup
  br label %ehcleanup21

ehcleanup21:                                      ; preds = %lpad15, %ehcleanup
  %actual_other.sroa.0.0 = phi i64 [ %1, %ehcleanup ], [ %4, %lpad15 ]
  %8 = inttoptr i64 %actual_other.sroa.0.0 to %C*
  br i1 undef, label %_ZN4CGAL6HandleD2Ev.exit, label %land.lhs.true.i

land.lhs.true.i:                                  ; preds = %ehcleanup21
  %count.i = getelementptr inbounds %C, %C* %8, i64 0, i32 1
  %9 = load i32, i32* %count.i, align 8
  unreachable

_ZN4CGAL6HandleD2Ev.exit:                         ; preds = %ehcleanup21
  resume { i8*, i32 } undef
}

; Function Attrs: nobuiltin
declare noalias nonnull i8* @_Znwm() local_unnamed_addr #1

; Function Attrs: sanitize_memory uwtable
declare void @scale() local_unnamed_addr #2 align 2

; Function Attrs: sanitize_memory uwtable
declare void @lazy() unnamed_addr #2 align 2

attributes #0 = { inlinehint sanitize_memory uwtable}
attributes #1 = { nobuiltin } 
attributes #2 = { sanitize_memory uwtable } 
attributes #3 = { builtin }

