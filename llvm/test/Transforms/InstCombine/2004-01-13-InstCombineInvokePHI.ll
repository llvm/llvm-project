; Test for a problem afflicting several C++ programs in the testsuite.  The 
; instcombine pass is trying to get rid of the cast in the invoke instruction, 
; inserting a cast of the return value after the PHI instruction, but which is
; used by the PHI instruction.  This is bad: because of the semantics of the
; invoke instruction, we really cannot perform this transformation at all at
; least without splitting the critical edge.
;
; RUN: opt < %s -instcombine -disable-output

declare i8* @test()

define i32 @foo() personality i32 (...)* @__gxx_personality_v0 {
entry:
        br i1 true, label %cont, label %call

call:           ; preds = %entry
        %P = invoke i32* bitcast (i8* ()* @test to i32* ()*)( )
                        to label %cont unwind label %N          ; <i32*> [#uses=1]

cont:           ; preds = %call, %entry
        %P2 = phi i32* [ %P, %call ], [ null, %entry ]          ; <i32*> [#uses=1]
        %V = load i32, i32* %P2              ; <i32> [#uses=1]
        ret i32 %V

N:              ; preds = %call
        %exn = landingpad {i8*, i32}
                 cleanup
        ret i32 0
}

declare i32 @__gxx_personality_v0(...)
