; RUN: llc < %s -O0 -relocation-model=pic -disable-fp-elim
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:64:64-v128:128:128-a0:0:64-n32"
target triple = "armv6-apple-darwin10"

%struct0 = type { i32, i32 }

; This function would crash RegAllocFast because it tried to spill %CPSR.
define arm_apcscc void @clobber_cc() nounwind noinline ssp {
entry:
  %asmtmp = call %struct0 asm sideeffect "...", "=&r,=&r,r,Ir,r,~{cc},~{memory}"(i32* undef, i32 undef, i32 1) nounwind ; <%0> [#uses=0]
  unreachable
}

@.str523 = private constant [256 x i8] c"<Unknown>\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00", align 4 ; <[256 x i8]*> [#uses=1]
declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) nounwind

; This function uses the scavenger for an ADDri instruction.
; ARMBaseRegisterInfo::estimateRSStackSizeLimit must return a 255 limit.
define arm_apcscc void @scavence_ADDri() nounwind {
entry:
  %letter = alloca i8                             ; <i8*> [#uses=0]
  %prodvers = alloca [256 x i8]                   ; <[256 x i8]*> [#uses=1]
  %buildver = alloca [256 x i8]                   ; <[256 x i8]*> [#uses=0]
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* undef, i8* getelementptr inbounds ([256 x i8]* @.str523, i32 0, i32 0), i32 256, i32 1, i1 false)
  %prodvers2 = bitcast [256 x i8]* %prodvers to i8* ; <i8*> [#uses=1]
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %prodvers2, i8* getelementptr inbounds ([256 x i8]* @.str523, i32 0, i32 0), i32 256, i32 1, i1 false)
  unreachable
}
