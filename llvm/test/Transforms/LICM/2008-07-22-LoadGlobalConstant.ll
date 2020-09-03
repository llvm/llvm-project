; RUN: opt < %s -basic-aa -licm -S | FileCheck %s

@a = external constant float*

define void @test(i32 %count) {
entry:
        br label %forcond

; CHECK:  %tmp3 = load float*, float** @a
; CHECK:  br label %forcond

forcond:
        %i.0 = phi i32 [ 0, %entry ], [ %inc, %forbody ]
        %cmp = icmp ult i32 %i.0, %count
        br i1 %cmp, label %forbody, label %afterfor

; CHECK:  %i.0 = phi i32 [ 0, %entry ], [ %inc, %forbody ]
; CHECK:  %cmp = icmp ult i32 %i.0, %count
; CHECK:  br i1 %cmp, label %forbody, label %afterfor

forbody:
        %tmp3 = load float*, float** @a
        %arrayidx = getelementptr float, float* %tmp3, i32 %i.0
        %tmp7 = uitofp i32 %i.0 to float
        store float %tmp7, float* %arrayidx
        %inc = add i32 %i.0, 1
        br label %forcond

; CHECK:  %arrayidx = getelementptr float, float* %tmp3, i32 %i.0
; CHECK:  %tmp7 = uitofp i32 %i.0 to float
; CHECK:  store float %tmp7, float* %arrayidx
; CHECK:  %inc = add i32 %i.0, 1
; CHECK:  br label %forcond

afterfor:
        ret void
}

; CHECK:  ret void
