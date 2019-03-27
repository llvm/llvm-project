; Test that VEXTEND or VROUND nodes are not emitted without vector support.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 | FileCheck %s

; CHECK-LABEL: fun1:
; CHECK: ldeb
; CHECK-LABEL: fun2:
; CHECK: ledbr

@.str = external dso_local unnamed_addr constant [21 x i8], align 2

define void @fun1() #0 {
bb:
%tmp = load <4 x float>, <4 x float>* undef, align 16
%tmp1 = extractelement <4 x float> %tmp, i32 0
%tmp2 = fpext float %tmp1 to double
%tmp3 = extractelement <4 x float> %tmp, i32 2
%tmp4 = fpext float %tmp3 to double
tail call void (i8*, ...) @printf(i8* getelementptr inbounds ([21 x i8], [21 x i8]* @.str, i64 0, i64 0), double %tmp2, double undef, double %tmp4, double undef)
ret void
}

define void @fun2() #0 {
bb:
%tmp = load <2 x double>, <2 x double>* undef, align 16
%tmp1 = extractelement <2 x double> %tmp, i32 0
%tmp2 = fptrunc double %tmp1 to float
%tmp3 = extractelement <2 x double> %tmp, i32 1
%tmp4 = fptrunc double %tmp3 to float
tail call void (i8*, ...) @printf(i8* getelementptr inbounds ([21 x i8], [21 x i8]* @.str, i64 0, i64 0), float %tmp2, float undef, float %tmp4, float undef)
ret void
}

declare dso_local void @printf(i8*, ...) #0
