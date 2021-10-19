; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu < %s | FileCheck %s

; Paramater Save Area is not needed if number of parameter does not exceed
; number of registers
; ------------------------------------------------------------------------------

; Max number of GPR is 8
define linkonce_odr void
    @WithoutParamArea(i8* %a, i32 signext %b) align 2 {
entry:
  call fastcc void @fastccFunc(i32 signext 1)
  ret void

; CHECK-LABEL: WithoutParamArea
; CHECK: stdu 1, -32(1)
; CHECK: blr
}
declare fastcc void @fastccFunc(i32 signext %level) unnamed_addr

; No need for Parameter Save Area if only 8 GPRs is needed.
define linkonce_odr void @WithoutParamArea2(i8* %a, i32 signext %b) align 2 {
entry:
  call fastcc void @eightArgs(i32 signext 1, i32 signext 2, i32 signext 3,
                              i32 signext 4, i32 signext 5, i32 signext 6,
                              i32 signext 7, i32 signext 8)
  ret void

; CHECK-LABEL: WithoutParamArea2
; CHECK: stdu 1, -32(1)
; CHECK: blr
}

declare fastcc void
    @eightArgs(i32 signext %level, i32 signext %level2, i32 signext %level3,
               i32 signext %level4, i32 signext %level5, i32 signext %level6,
               i32 signext %level7, i32 signext %level8) unnamed_addr

; No need for Parameter Save Area for calls that utiliizes 8 GPR and 2 FPR.
define linkonce_odr void @WithoutParamArea3(i8* %a, i32 signext %b) align 2 {
entry:
  call fastcc void
      @mixedArgs(i32 signext 1, float 1.0, i32 signext 2, float 2.0,
                 i32 signext 3, i32 signext 4, i32 signext 5, i32 signext 6,
                 i32 signext 7, i32 signext 8) ret void
  ret void

; CHECK-LABEL: WithoutParamArea3
; CHECK: stdu 1, -32(1)
; CHECK: blr
}

declare fastcc void
    @mixedArgs(i32 signext %level, float %levelf1, i32 signext %level2,
               float %levelf2, i32 signext %level3, i32 signext %level4,
               i32 signext %level5, i32 signext %level6, i32 signext %level7,
               i32 signext %level8) unnamed_addr

; Pass by value usage requiring less GPR then available
%"myClass::Mem" = type { i8, i8, i16, i32, i32, i32, i64 }

define internal fastcc void @CallPassByValue(%"myClass::Mem"* %E) align 2 {
entry:
  call fastcc void @PassByValue(%"myClass::Mem"* byval(%"myClass::Mem") nonnull align 8 undef);
  ret void

; CHECK-LABEL: PassByValue
; CHECK: stdu 1, -32(1)
; CHECK: blr
}

declare dso_local fastcc void
    @PassByValue(%"myClass::Mem"* byval(%"myClass::Mem") nocapture readonly align 8) align 2

; Verify Paramater Save Area is allocated if parameter exceed the number that
; can be passed via registers
; ------------------------------------------------------------------------------

; Max number of GPR is 8
define linkonce_odr void @WithParamArea(i8 * %a, i32 signext %b) align 2 {
entry:
  call fastcc void @nineArgs(i32 signext 1, i32 signext 2, i32 signext 3,
                           i32 signext 4, i32 signext 5, i32 signext 6,
                           i32 signext 7, i32 signext 8, i32 signext 9)
  ret void

; CHECK-LABEL: WithParamArea
; CHECK: stdu 1, -96(1)
; CHECK: blr
}

declare fastcc void @nineArgs(i32 signext %level, i32 signext %level2,
  i32 signext %level3, i32 signext %level4, i32 signext %level5,
  i32 signext %level6, i32 signext %level7, i32 signext %level8,
  i32 signext %level9) unnamed_addr

; Max number of FPR for parameter passing is 13
define linkonce_odr void @WithParamArea2(i8* %a, i32 signext %b) align 2 {
entry:
  call fastcc void @funcW14FloatArgs(float 1.0, float 2.0, float 3.0,
    float 4.0, float 5.0, float 6.0, float 7.0, float 8.0, float 1.0,
    float 2.0, float 3.0, float 4.0, float 5.0, float 14.0)
  ret void

; CHECK-LABEL: WithParamArea2
; CHECK: stdu 1, -96(1)
; CHECK: blr
}

declare fastcc void
    @funcW14FloatArgs(float %level, float %level2, float %level3,
                      float %level4, float %level5, float %level6,
                      float %level7, float %level8, float %level9,
                      float %level10, float %level11, float %level12,
                      float %level13, float %level14);


; Pass by value usage requires more GPR then available
%"myClass::MemA" = type { i8, i8, i16, i32, i32, i32, i64 }
%"myClass::MemB" = type { i32*, i32, i32, %"myClass::MemB"** }
%"myClass::MemC" = type { %"myClass::MemD"*, %"myClass::MemC"*, i64 }
%"myClass::MemD" = type { %"myClass::MemB"*, %"myClass::MemC"*, i8, i8, i16,
              i32 }
%"myStruct::MemF" = type { i32, %"myClass::MemA"*, %"myClass::MemA"*, i64, i64 }
%"myClass::MemK" = type { i32, %"myClass::MemD"*, %"myClass::MemD"*, i64, i32,
                          i64, i8, i32, %"myStruct::MemF",
              i8, %"myClass::MemA"* }

define internal fastcc void @AggMemExprEmitter(%"myClass::MemK"* %E) align 2 {
entry:
  call fastcc void @MemExprEmitterInitialization(%"myClass::MemK"*
                                               byval(%"myClass::MemK") nonnull align 8 undef);
  ret void

; CHECK-LABEL: AggMemExprEmitter
; CHECK: stdu 1, -144(1)
; CHECK: blr
}

declare dso_local fastcc void
    @MemExprEmitterInitialization(%"myClass::MemK"*
                                  byval(%"myClass::MemK") nocapture readonly align 8) align 2
