; RUN: llc < %s -mcpu=yonah -relocation-model=static | grep movaps | count 2

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i686-apple-darwin8"

%f4 = type <4 x float>

@G = external global { float,float,float,float}, align 16

define %f4 @test1(float %W, float %X, float %Y, float %Z) nounwind {
        %tmp = insertelement %f4 undef, float %W, i32 0
        %tmp2 = insertelement %f4 %tmp, float %X, i32 1
        %tmp4 = insertelement %f4 %tmp2, float %Y, i32 2
        %tmp6 = insertelement %f4 %tmp4, float %Z, i32 3
	ret %f4 %tmp6
}

define %f4 @test2() nounwind {
	%Wp = getelementptr { float,float,float,float}* @G, i32 0, i32 0
	%Xp = getelementptr { float,float,float,float}* @G, i32 0, i32 1
	%Yp = getelementptr { float,float,float,float}* @G, i32 0, i32 2
	%Zp = getelementptr { float,float,float,float}* @G, i32 0, i32 3
	
	%W = load float* %Wp
	%X = load float* %Xp
	%Y = load float* %Yp
	%Z = load float* %Zp

        %tmp = insertelement %f4 undef, float %W, i32 0
        %tmp2 = insertelement %f4 %tmp, float %X, i32 1
        %tmp4 = insertelement %f4 %tmp2, float %Y, i32 2
        %tmp6 = insertelement %f4 %tmp4, float %Z, i32 3
	ret %f4 %tmp6
}

