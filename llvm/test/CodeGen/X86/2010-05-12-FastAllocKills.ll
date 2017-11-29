; RUN: llc -regalloc=fast -optimize-regalloc=0 -verify-machineinstrs < %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin"

; This test causes a virtual FP register to be redefined while it is live:
;BB#5: derived from LLVM BB %bb10
;    Predecessors according to CFG: BB#4 BB#5
;	%reg1024<def> = MOV_Fp8080 %reg1034
;	%reg1025<def> = MUL_Fp80m32 %reg1024, %rip, 1, %reg0, <cp#0>, %reg0; mem:LD4[ConstantPool]
;	%reg1034<def> = MOV_Fp8080 %reg1025
;	FP_REG_KILL %fp0<imp-def>, %fp1<imp-def>, %fp2<imp-def>, %fp3<imp-def>, %fp4<imp-def>, %fp5<imp-def>, %fp6<imp-def>
;	JMP_4 <BB#5>
;    Successors according to CFG: BB#5
;
; The X86FP pass needs good kill flags, like on %fp0 representing %reg1034:
;BB#5: derived from LLVM BB %bb10
;    Predecessors according to CFG: BB#4 BB#5
;	%fp0<def> = LD_Fp80m <fi#3>, 1, %reg0, 0, %reg0; mem:LD10[FixedStack3](align=4)
;	%fp1<def> = MOV_Fp8080 %fp0<kill>
;	%fp2<def> = MUL_Fp80m32 %fp1, %rip, 1, %reg0, <cp#0>, %reg0; mem:LD4[ConstantPool]
;	%fp0<def> = MOV_Fp8080 %fp2
;	ST_FpP80m <fi#3>, 1, %reg0, 0, %reg0, %fp0<kill>; mem:ST10[FixedStack3](align=4)
;	ST_FpP80m <fi#4>, 1, %reg0, 0, %reg0, %fp1<kill>; mem:ST10[FixedStack4](align=4)
;	ST_FpP80m <fi#5>, 1, %reg0, 0, %reg0, %fp2<kill>; mem:ST10[FixedStack5](align=4)
;	FP_REG_KILL %fp0<imp-def>, %fp1<imp-def>, %fp2<imp-def>, %fp3<imp-def>, %fp4<imp-def>, %fp5<imp-def>, %fp6<imp-def>
;	JMP_4 <BB#5>
;    Successors according to CFG: BB#5

define fastcc i32 @sqlite3AtoF(i8* %z, double* nocapture %pResult) nounwind ssp {
entry:
  br i1 undef, label %bb2, label %bb1.i.i

bb1.i.i:                                          ; preds = %entry
  unreachable

bb2:                                              ; preds = %entry
  br i1 undef, label %isdigit339.exit11.preheader, label %bb13

isdigit339.exit11.preheader:                      ; preds = %bb2
  br i1 undef, label %bb12, label %bb10

bb10:                                             ; preds = %bb10, %isdigit339.exit11.preheader
  %divisor.041 = phi x86_fp80 [ %0, %bb10 ], [ 0xK3FFF8000000000000000, %isdigit339.exit11.preheader ] ; <x86_fp80> [#uses=1]
  %0 = fmul x86_fp80 %divisor.041, 0xK4002A000000000000000 ; <x86_fp80> [#uses=2]
  br i1 false, label %bb12, label %bb10

bb12:                                             ; preds = %bb10, %isdigit339.exit11.preheader
  %divisor.0.lcssa = phi x86_fp80 [ 0xK3FFF8000000000000000, %isdigit339.exit11.preheader ], [ %0, %bb10 ] ; <x86_fp80> [#uses=0]
  br label %bb13

bb13:                                             ; preds = %bb12, %bb2
  br i1 undef, label %bb34, label %bb36

bb34:                                             ; preds = %bb13
  br label %bb36

bb36:                                             ; preds = %bb34, %bb13
  ret i32 undef
}
