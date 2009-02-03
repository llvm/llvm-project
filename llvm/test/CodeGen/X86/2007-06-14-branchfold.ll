; RUN: llvm-as < %s | llc -march=x86 -mcpu=i686 | not grep jmp
; check that branch folding understands FP_REG_KILL is not a branch

target triple = "i686-pc-linux-gnu"
  %struct.FRAME.c34003a = type { float, float }
@report_E = global i8 0   ; <i8*> [#uses=0]

define void @main() {
entry:
  %FRAME.31 = alloca %struct.FRAME.c34003a, align 8   ; <%struct.FRAME.c34003a*> [#uses=4]
  %tmp20 = call i32 @report__ident_int( i32 -50 )   ; <i32> [#uses=1]
  %tmp2021 = sitofp i32 %tmp20 to float   ; <float> [#uses=5]
  %tmp23 = fcmp ult float %tmp2021, 0xC7EFFFFFE0000000    ; <i1> [#uses=1]
  %tmp26 = fcmp ugt float %tmp2021, 0x47EFFFFFE0000000    ; <i1> [#uses=1]
  %bothcond = or i1 %tmp23, %tmp26    ; <i1> [#uses=1]
  br i1 %bothcond, label %bb, label %bb30

bb:   ; preds = %entry
  unwind

bb30:   ; preds = %entry
  %tmp35 = call i32 @report__ident_int( i32 50 )    ; <i32> [#uses=1]
  %tmp3536 = sitofp i32 %tmp35 to float   ; <float> [#uses=4]
  %tmp38 = fcmp ult float %tmp3536, 0xC7EFFFFFE0000000    ; <i1> [#uses=1]
  %tmp44 = fcmp ugt float %tmp3536, 0x47EFFFFFE0000000    ; <i1> [#uses=1]
  %bothcond226 = or i1 %tmp38, %tmp44   ; <i1> [#uses=1]
  br i1 %bothcond226, label %bb47, label %bb49

bb47:   ; preds = %bb30
  unwind

bb49:   ; preds = %bb30
  %tmp60 = fcmp ult float %tmp3536, %tmp2021    ; <i1> [#uses=1]
  %tmp60.not = xor i1 %tmp60, true    ; <i1> [#uses=1]
  %tmp65 = fcmp olt float %tmp2021, 0xC7EFFFFFE0000000    ; <i1> [#uses=1]
  %bothcond227 = and i1 %tmp65, %tmp60.not    ; <i1> [#uses=1]
  br i1 %bothcond227, label %cond_true68, label %cond_next70

cond_true68:    ; preds = %bb49
  unwind

cond_next70:    ; preds = %bb49
  %tmp71 = call i32 @report__ident_int( i32 -30 )   ; <i32> [#uses=1]
  %tmp7172 = sitofp i32 %tmp71 to float   ; <float> [#uses=3]
  %tmp74 = fcmp ult float %tmp7172, 0xC7EFFFFFE0000000    ; <i1> [#uses=1]
  %tmp80 = fcmp ugt float %tmp7172, 0x47EFFFFFE0000000    ; <i1> [#uses=1]
  %bothcond228 = or i1 %tmp74, %tmp80   ; <i1> [#uses=1]
  br i1 %bothcond228, label %bb83, label %bb85

bb83:   ; preds = %cond_next70
  unwind

bb85:   ; preds = %cond_next70
  %tmp90 = getelementptr %struct.FRAME.c34003a* %FRAME.31, i32 0, i32 1   ; <float*> [#uses=3]
  store float %tmp7172, float* %tmp90
  %tmp92 = call i32 @report__ident_int( i32 30 )    ; <i32> [#uses=1]
  %tmp9293 = sitofp i32 %tmp92 to float   ; <float> [#uses=7]
  %tmp95 = fcmp ult float %tmp9293, 0xC7EFFFFFE0000000    ; <i1> [#uses=1]
  %tmp101 = fcmp ugt float %tmp9293, 0x47EFFFFFE0000000   ; <i1> [#uses=1]
  %bothcond229 = or i1 %tmp95, %tmp101    ; <i1> [#uses=1]
  br i1 %bothcond229, label %bb104, label %bb106

bb104:    ; preds = %bb85
  unwind

bb106:    ; preds = %bb85
  %tmp111 = getelementptr %struct.FRAME.c34003a* %FRAME.31, i32 0, i32 0    ; <float*> [#uses=2]
  store float %tmp9293, float* %tmp111
  %tmp123 = load float* %tmp90    ; <float> [#uses=4]
  %tmp125 = fcmp ult float %tmp9293, %tmp123    ; <i1> [#uses=1]
  br i1 %tmp125, label %cond_next147, label %cond_true128

cond_true128:   ; preds = %bb106
  %tmp133 = fcmp olt float %tmp123, %tmp2021    ; <i1> [#uses=1]
  %tmp142 = fcmp ogt float %tmp9293, %tmp3536   ; <i1> [#uses=1]
  %bothcond230 = or i1 %tmp133, %tmp142   ; <i1> [#uses=1]
  br i1 %bothcond230, label %bb145, label %cond_next147

bb145:    ; preds = %cond_true128
  unwind

cond_next147:   ; preds = %cond_true128, %bb106
  %tmp157 = fcmp ugt float %tmp123, -3.000000e+01   ; <i1> [#uses=1]
  %tmp165 = fcmp ult float %tmp9293, -3.000000e+01    ; <i1> [#uses=1]
  %bothcond231 = or i1 %tmp157, %tmp165   ; <i1> [#uses=1]
  br i1 %bothcond231, label %bb168, label %bb169

bb168:    ; preds = %cond_next147
  unwind

bb169:    ; preds = %cond_next147
  %tmp176 = fcmp ugt float %tmp123, 3.000000e+01    ; <i1> [#uses=1]
  %tmp184 = fcmp ult float %tmp9293, 3.000000e+01   ; <i1> [#uses=1]
  %bothcond232 = or i1 %tmp176, %tmp184   ; <i1> [#uses=1]
  br i1 %bothcond232, label %bb187, label %bb188

bb187:    ; preds = %bb169
  unwind

bb188:    ; preds = %bb169
  %tmp192 = call fastcc float @c34003a__ident.154( %struct.FRAME.c34003a* %FRAME.31, float 3.000000e+01 )   ; <float> [#uses=2]
  %tmp194 = load float* %tmp90    ; <float> [#uses=1]
  %tmp196 = fcmp ugt float %tmp194, 0.000000e+00    ; <i1> [#uses=1]
  br i1 %tmp196, label %bb207, label %cond_next200

cond_next200:   ; preds = %bb188
  %tmp202 = load float* %tmp111   ; <float> [#uses=1]
  %tmp204 = fcmp ult float %tmp202, 0.000000e+00    ; <i1> [#uses=1]
  br i1 %tmp204, label %bb207, label %bb208

bb207:    ; preds = %cond_next200, %bb188
  unwind

bb208:    ; preds = %cond_next200
  %tmp212 = call fastcc float @c34003a__ident.154( %struct.FRAME.c34003a* %FRAME.31, float 0.000000e+00 )   ; <float> [#uses=1]
  %tmp214 = fcmp oge float %tmp212, %tmp192   ; <i1> [#uses=1]
  %tmp217 = fcmp oge float %tmp192, 1.000000e+02    ; <i1> [#uses=1]
  %tmp221 = or i1 %tmp214, %tmp217    ; <i1> [#uses=1]
  br i1 %tmp221, label %cond_true224, label %UnifiedReturnBlock

cond_true224:   ; preds = %bb208
  call void @abort( ) noreturn
  ret void

UnifiedReturnBlock:   ; preds = %bb208
  ret void
}

declare fastcc float @c34003a__ident.154(%struct.FRAME.c34003a* %CHAIN.32, float %x) 

declare i32 @report__ident_int(i32 %x)

declare void @abort() noreturn
