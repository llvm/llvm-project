; RUN: opt < %s -passes=adce -disable-output

declare void @strlen()

declare void @_ZN10QByteArray6resizeEi()

declare void @q_atomic_decrement()

define void @_ZNK10QByteArray13leftJustifiedEicb() personality ptr @__gxx_personality_v0 {
entry:
        invoke void @strlen( )
                        to label %tmp.3.i.noexc unwind label %invoke_catch.0

tmp.3.i.noexc:          ; preds = %entry
        br i1 false, label %then.0, label %else.0

invoke_catch.0:         ; preds = %entry
        %exn.0 = landingpad {ptr, i32}
                 cleanup
        invoke void @q_atomic_decrement( )
                        to label %tmp.1.i.i183.noexc unwind label %terminate

tmp.1.i.i183.noexc:             ; preds = %invoke_catch.0
        ret void

then.0:         ; preds = %tmp.3.i.noexc
        invoke void @_ZN10QByteArray6resizeEi( )
                        to label %invoke_cont.1 unwind label %invoke_catch.1

invoke_catch.1:         ; preds = %then.0
        %exn.1 = landingpad {ptr, i32}
                 cleanup
        invoke void @q_atomic_decrement( )
                        to label %tmp.1.i.i162.noexc unwind label %terminate

tmp.1.i.i162.noexc:             ; preds = %invoke_catch.1
        ret void

invoke_cont.1:          ; preds = %then.0
        ret void

else.0:         ; preds = %tmp.3.i.noexc
        ret void

terminate:              ; preds = %invoke_catch.1, %invoke_catch.0
        %dbg.0.1 = phi ptr [ null, %invoke_catch.1 ], [ null, %invoke_catch.0 ]               ; <ptr> [#uses=0]
        %exn = landingpad {ptr, i32}
                 cleanup
        unreachable
}

declare i32 @__gxx_personality_v0(...)
