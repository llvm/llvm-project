; RUN: llc < %s -O0 -relocation-model=pic -frame-pointer=all -no-integrated-as
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:64:64-v128:128:128-a0:0:64-n32"
target triple = "armv6-apple-darwin10"

%struct0 = type { i32, i32 }

; This function would crash RegAllocFast because it tried to spill %CPSR.
define arm_apcscc void @clobber_cc() nounwind noinline ssp {
entry:
  %asmtmp = call %struct0 asm sideeffect "...", "=&r,=&r,r,Ir,r,~{cc},~{memory}"(ptr undef, i32 undef, i32 1) nounwind ; <%0> [#uses=0]
  unreachable
}

@.str523 = private constant [256 x i8] c"<Unknown>\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00", align 4 ; <ptr> [#uses=1]
declare void @llvm.memcpy.p0.p0.i32(ptr nocapture, ptr nocapture, i32, i1) nounwind

; This function uses the scavenger for an ADDri instruction.
; ARMBaseRegisterInfo::estimateRSStackSizeLimit must return a 255 limit.
define arm_apcscc void @scavence_ADDri() nounwind {
entry:
  %letter = alloca i8                             ; <ptr> [#uses=0]
  %prodvers = alloca [256 x i8]                   ; <ptr> [#uses=1]
  %buildver = alloca [256 x i8]                   ; <ptr> [#uses=0]
  call void @llvm.memcpy.p0.p0.i32(ptr align 1 undef, ptr align 1 @.str523, i32 256, i1 false)
  call void @llvm.memcpy.p0.p0.i32(ptr align 1 %prodvers, ptr align 1 @.str523, i32 256, i1 false)
  unreachable
}
