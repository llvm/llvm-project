; RUN: opt < %s -indvars -disable-output

define void @_ZN5ArrayISt7complexIdEEC2ERK10dim_vector() personality i32 (...)* @__gxx_personality_v0 {
entry:
        %tmp.7 = invoke i32 @_ZN5ArrayISt7complexIdEE8get_sizeERK10dim_vector( )
                        to label %invoke_cont.0 unwind label %cond_true.1               ; <i32> [#uses=2]

invoke_cont.0:          ; preds = %entry
        %tmp.4.i = bitcast i32 %tmp.7 to i32            ; <i32> [#uses=0]
        %tmp.14.0.i5 = add i32 %tmp.7, -1               ; <i32> [#uses=1]
        br label %no_exit.i

no_exit.i:              ; preds = %no_exit.i, %invoke_cont.0
        %tmp.14.0.i.0 = phi i32 [ %tmp.14.0.i, %no_exit.i ], [ %tmp.14.0.i5, %invoke_cont.0 ]           ; <i32> [#uses=1]
        %tmp.14.0.i = add i32 %tmp.14.0.i.0, -1         ; <i32> [#uses=1]
        br label %no_exit.i

cond_true.1:            ; preds = %entry
        %exn = landingpad {i8*, i32}
                 cleanup
        resume { i8*, i32 } %exn
}

declare i32 @__gxx_personality_v0(...)

declare i32 @_ZN5ArrayISt7complexIdEE8get_sizeERK10dim_vector()

