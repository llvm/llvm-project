; RUN: llvm-as < %s | opt -licm -enable-licm-constant-variables | llvm-dis | grep -A 1 entry | grep load.*@a
@a = external constant float*

define void @test(i32 %count) {
entry:
        br label %forcond

forcond:
        %i.0 = phi i32 [ 0, %entry ], [ %inc, %forbody ]
        %cmp = icmp ult i32 %i.0, %count
        br i1 %cmp, label %forbody, label %afterfor

forbody:
        %tmp3 = load float** @a
        %arrayidx = getelementptr float* %tmp3, i32 %i.0
        %tmp7 = uitofp i32 %i.0 to float
        store float %tmp7, float* %arrayidx
        %inc = add i32 %i.0, 1
        br label %forcond

afterfor:
        ret void
}
