; RUN: opt < %s -indvars -S | FileCheck %s
; PR4086
declare void @foo()

define void @test() {
entry:
        br label %loop_body

loop_body:
        %i = phi float [ %nexti, %loop_body ], [ 0.0, %entry ]
        tail call void @foo()
        %nexti = fadd float %i, 1.0
        ; CHECK: icmp ne i32 %{{[a-zA-Z$._0-9]+}}, 2
        %less = fcmp olt float %nexti, 2.0
        br i1 %less, label %loop_body, label %done

done:
        ret void
}
