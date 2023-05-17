; Test for a problem afflicting several C++ programs in the testsuite.  The 
; instcombine pass is trying to get rid of the cast in the invoke instruction, 
; inserting a cast of the return value after the PHI instruction, but which is
; used by the PHI instruction.  This is bad: because of the semantics of the
; invoke instruction, we really cannot perform this transformation at all at
; least without splitting the critical edge.
;
; RUN: opt < %s -passes=instcombine -disable-output

declare ptr @test()

define i32 @foo() personality ptr @__gxx_personality_v0 {
entry:
        br i1 true, label %cont, label %call

call:           ; preds = %entry
        %P = invoke ptr @test( )
                        to label %cont unwind label %N          ; <ptr> [#uses=1]

cont:           ; preds = %call, %entry
        %P2 = phi ptr [ %P, %call ], [ null, %entry ]          ; <ptr> [#uses=1]
        %V = load i32, ptr %P2              ; <i32> [#uses=1]
        ret i32 %V

N:              ; preds = %call
        %exn = landingpad {ptr, i32}
                 cleanup
        ret i32 0
}

declare i32 @__gxx_personality_v0(...)
