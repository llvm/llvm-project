; RUN: llc -mtriple=spirv-unknown-vulkan-compute -O0 %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val %}

;
; int foo() { return 200; }
;
; [numthreads(1, 1, 1)]
; void main() {
;   int result;
;
;   ////////////////////////////
;   // The most basic case    //
;   // Has a 'default' case   //
;   // All cases have 'break' //
;   ////////////////////////////
;   int a = 0;
;   switch(a) {
;     case -3:
;       result = -300;
;       break;
;     case 0:
;       result = 0;
;       break;
;     case 1:
;       result = 100;
;       break;
;     case 2:
;       result = foo();
;       break;
;     default:
;       result = 777;
;       break;
;   }
;
;   ////////////////////////////////////
;   // The selector is a statement    //
;   // Does not have a 'default' case //
;   // All cases have 'break'         //
;   ////////////////////////////////////
;
;   switch(int c = a) {
;     case -4:
;       result = -400;
;       break;
;     case 4:
;       result = 400;
;       break;
;   }
;
;   ///////////////////////////////////
;   // All cases are fall-through    //
;   // The last case is fall-through //
;   ///////////////////////////////////
;   switch(a) {
;     case -5:
;       result = -500;
;     case 5:
;       result = 500;
;   }
;
;   ///////////////////////////////////////
;   // Some cases are fall-through       //
;   // The last case is not fall-through //
;   ///////////////////////////////////////
;
;   switch(a) {
;     case 6:
;       result = 600;
;     case 7:
;       result = 700;
;     case 8:
;       result = 800;
;       break;
;     default:
;       result = 777;
;       break;
;   }
;
;   ///////////////////////////////////////
;   // Fall-through cases with no body   //
;   ///////////////////////////////////////
;
;   switch(a) {
;     case 10:
;     case 11:
;     default:
;     case 12:
;       result = 12;
;   }
;
;   ////////////////////////////////////////////////
;   // No-op. Two nested cases and a nested break //
;   ////////////////////////////////////////////////
;
;   switch(a) {
;     case 15:
;     case 16:
;       break;
;   }
;
;   ////////////////////////////////////////////////////////////////
;   // Using braces (compound statements) in various parts        //
;   // Using breaks such that each AST configuration is different //
;   // Also uses 'forcecase' attribute                            //
;   ////////////////////////////////////////////////////////////////
;
;   switch(a) {
;     case 20: {
;       result = 20;
;       break;
;     }
;     case 21:
;       result = 21;
;       break;
;     case 22:
;     case 23:
;       break;
;     case 24:
;     case 25: { result = 25; }
;       break;
;     case 26:
;     case 27: {
;       break;
;     }
;     case 28: {
;       result = 28;
;       {{break;}}
;     }
;     case 29: {
;       {
;         result = 29;
;         {break;}
;       }
;     }
;   }
;
;   ////////////////////////////////////////////////////////////////////////
;   // Nested Switch statements with mixed use of fall-through and braces //
;   ////////////////////////////////////////////////////////////////////////
;
;   switch(a) {
;     case 30: {
;         result = 30;
;         switch(result) {
;           default:
;             a = 55;
;           case 50:
;             a = 50;
;             break;
;           case 51:
;           case 52:
;             a = 52;
;           case 53:
;             a = 53;
;             break;
;           case 54 : {
;             a = 54;
;             break;
;           }
;         }
;     }
;   }
;
;   ///////////////////////////////////////////////
;   // Constant integer variables as case values //
;   ///////////////////////////////////////////////
;
;   const int r = 35;
;   const int s = 45;
;   const int t = 2*r + s;  // evaluates to 115.
;
;   switch(a) {
;     case r:
;       result = r;
;     case t:
;       result = t;
;       break;
;   }
;
;
;   //////////////////////////////////////////////////////////////////
;   // Using float as selector results in multiple casts in the AST //
;   //////////////////////////////////////////////////////////////////
;   float sel;
;   switch ((int)sel) {
;   case 0:
;     result = 0;
;     break;
;   }
; }

; CHECK: %[[#func_41:]] = OpFunction %[[#uint:]] DontInline %[[#]]
; CHECK:    %[[#bb82:]] = OpLabel
; CHECK:                  OpReturnValue %[[#]]
; CHECK:                  OpFunctionEnd
; CHECK: %[[#func_42:]] = OpFunction %[[#void:]] DontInline %[[#]]
; CHECK:    %[[#bb83:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb84:]] None
; CHECK:                  OpSwitch %[[#]] %[[#bb85:]] 4294967293 %[[#bb86:]] 0 %[[#bb87:]] 1 %[[#bb88:]] 2 %[[#bb89:]]
; CHECK:    %[[#bb85:]] = OpLabel
; CHECK:                  OpBranch %[[#bb84:]]
; CHECK:    %[[#bb86:]] = OpLabel
; CHECK:                  OpBranch %[[#bb84:]]
; CHECK:    %[[#bb87:]] = OpLabel
; CHECK:                  OpBranch %[[#bb84:]]
; CHECK:    %[[#bb88:]] = OpLabel
; CHECK:                  OpBranch %[[#bb84:]]
; CHECK:    %[[#bb89:]] = OpLabel
; CHECK:                  OpBranch %[[#bb84:]]
; CHECK:    %[[#bb84:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb90:]] None
; CHECK:                  OpSwitch %[[#]] %[[#bb90:]] 4294967292 %[[#bb91:]] 4 %[[#bb92:]]
; CHECK:    %[[#bb91:]] = OpLabel
; CHECK:                  OpBranch %[[#bb90:]]
; CHECK:    %[[#bb92:]] = OpLabel
; CHECK:                  OpBranch %[[#bb90:]]
; CHECK:    %[[#bb90:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb93:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb94:]] %[[#bb95:]]
; CHECK:    %[[#bb94:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb96:]] None
; CHECK:                  OpSwitch %[[#]] %[[#bb96:]] 4294967291 %[[#bb97:]] 5 %[[#bb98:]]
; CHECK:    %[[#bb95:]] = OpLabel
; CHECK:    %[[#bb97:]] = OpLabel
; CHECK:                  OpBranch %[[#bb96:]]
; CHECK:    %[[#bb98:]] = OpLabel
; CHECK:                  OpBranch %[[#bb96:]]
; CHECK:    %[[#bb96:]] = OpLabel
; CHECK:                  OpBranchConditional %[[#]] %[[#bb93:]] %[[#bb99:]]
; CHECK:    %[[#bb99:]] = OpLabel
; CHECK:                  OpBranch %[[#bb93:]]
; CHECK:    %[[#bb93:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb100:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb101:]] %[[#bb102:]]
; CHECK:   %[[#bb101:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb103:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb104:]] %[[#bb105:]]
; CHECK:   %[[#bb102:]] = OpLabel
; CHECK:   %[[#bb104:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb106:]] None
; CHECK:                  OpSwitch %[[#]] %[[#bb107:]] 6 %[[#bb108:]] 7 %[[#bb106:]] 8 %[[#bb109:]]
; CHECK:   %[[#bb105:]] = OpLabel
; CHECK:   %[[#bb107:]] = OpLabel
; CHECK:                  OpBranch %[[#bb106:]]
; CHECK:   %[[#bb108:]] = OpLabel
; CHECK:                  OpBranch %[[#bb106:]]
; CHECK:   %[[#bb109:]] = OpLabel
; CHECK:                  OpBranch %[[#bb106:]]
; CHECK:   %[[#bb106:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb110:]] None
; CHECK:                  OpSwitch %[[#]] %[[#bb111:]] 1 %[[#bb110:]] 2 %[[#bb112:]]
; CHECK:   %[[#bb111:]] = OpLabel
; CHECK:                  OpBranch %[[#bb110:]]
; CHECK:   %[[#bb112:]] = OpLabel
; CHECK:                  OpBranch %[[#bb110:]]
; CHECK:   %[[#bb110:]] = OpLabel
; CHECK:                  OpBranch %[[#bb103:]]
; CHECK:   %[[#bb103:]] = OpLabel
; CHECK:                  OpBranchConditional %[[#]] %[[#bb113:]] %[[#bb100:]]
; CHECK:   %[[#bb113:]] = OpLabel
; CHECK:                  OpBranch %[[#bb100:]]
; CHECK:   %[[#bb100:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb114:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb115:]] %[[#bb116:]]
; CHECK:   %[[#bb115:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb117:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb118:]] %[[#bb119:]]
; CHECK:   %[[#bb116:]] = OpLabel
; CHECK:   %[[#bb118:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb120:]] None
; CHECK:                  OpSwitch %[[#]] %[[#bb120:]] 10 %[[#bb121:]] 11 %[[#bb122:]] 12 %[[#bb123:]]
; CHECK:   %[[#bb119:]] = OpLabel
; CHECK:   %[[#bb121:]] = OpLabel
; CHECK:                  OpBranch %[[#bb120:]]
; CHECK:   %[[#bb122:]] = OpLabel
; CHECK:                  OpBranch %[[#bb120:]]
; CHECK:   %[[#bb123:]] = OpLabel
; CHECK:                  OpBranch %[[#bb120:]]
; CHECK:   %[[#bb120:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb124:]] None
; CHECK:                  OpSwitch %[[#]] %[[#bb124:]] 1 %[[#bb125:]] 2 %[[#bb126:]]
; CHECK:   %[[#bb125:]] = OpLabel
; CHECK:                  OpBranch %[[#bb124:]]
; CHECK:   %[[#bb126:]] = OpLabel
; CHECK:                  OpBranch %[[#bb124:]]
; CHECK:   %[[#bb124:]] = OpLabel
; CHECK:                  OpBranch %[[#bb117:]]
; CHECK:   %[[#bb117:]] = OpLabel
; CHECK:                  OpBranch %[[#bb114:]]
; CHECK:   %[[#bb114:]] = OpLabel
; CHECK:                  OpBranch %[[#bb127:]]
; CHECK:   %[[#bb127:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb128:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb129:]] %[[#bb130:]]
; CHECK:   %[[#bb129:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb131:]] None
; CHECK:                  OpSwitch %[[#]] %[[#bb131:]] 15 %[[#bb132:]] 16 %[[#bb133:]]
; CHECK:   %[[#bb130:]] = OpLabel
; CHECK:   %[[#bb132:]] = OpLabel
; CHECK:                  OpBranch %[[#bb131:]]
; CHECK:   %[[#bb133:]] = OpLabel
; CHECK:                  OpBranch %[[#bb131:]]
; CHECK:   %[[#bb131:]] = OpLabel
; CHECK:                  OpBranch %[[#bb128:]]
; CHECK:   %[[#bb128:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb134:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb135:]] %[[#bb136:]]
; CHECK:   %[[#bb135:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb137:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb138:]] %[[#bb139:]]
; CHECK:   %[[#bb136:]] = OpLabel
; CHECK:   %[[#bb138:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb140:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb141:]] %[[#bb142:]]
; CHECK:   %[[#bb139:]] = OpLabel
; CHECK:   %[[#bb141:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb143:]] None
; CHECK:                  OpSwitch %[[#]] %[[#bb143:]] 20 %[[#bb144:]] 21 %[[#bb145:]] 22 %[[#bb146:]] 23 %[[#bb147:]] 24 %[[#bb148:]] 25 %[[#bb149:]] 26 %[[#bb150:]] 27 %[[#bb151:]] 28 %[[#bb152:]] 29 %[[#bb153:]]
; CHECK:   %[[#bb142:]] = OpLabel
; CHECK:   %[[#bb144:]] = OpLabel
; CHECK:                  OpBranch %[[#bb143:]]
; CHECK:   %[[#bb145:]] = OpLabel
; CHECK:                  OpBranch %[[#bb143:]]
; CHECK:   %[[#bb146:]] = OpLabel
; CHECK:                  OpBranch %[[#bb143:]]
; CHECK:   %[[#bb147:]] = OpLabel
; CHECK:                  OpBranch %[[#bb143:]]
; CHECK:   %[[#bb148:]] = OpLabel
; CHECK:                  OpBranch %[[#bb143:]]
; CHECK:   %[[#bb149:]] = OpLabel
; CHECK:                  OpBranch %[[#bb143:]]
; CHECK:   %[[#bb150:]] = OpLabel
; CHECK:                  OpBranch %[[#bb143:]]
; CHECK:   %[[#bb151:]] = OpLabel
; CHECK:                  OpBranch %[[#bb143:]]
; CHECK:   %[[#bb152:]] = OpLabel
; CHECK:                  OpBranch %[[#bb143:]]
; CHECK:   %[[#bb153:]] = OpLabel
; CHECK:                  OpBranch %[[#bb143:]]
; CHECK:   %[[#bb143:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb154:]] None
; CHECK:                  OpSwitch %[[#]] %[[#bb154:]] 1 %[[#bb155:]] 2 %[[#bb156:]] 3 %[[#bb157:]]
; CHECK:   %[[#bb155:]] = OpLabel
; CHECK:                  OpBranch %[[#bb154:]]
; CHECK:   %[[#bb156:]] = OpLabel
; CHECK:                  OpBranch %[[#bb154:]]
; CHECK:   %[[#bb157:]] = OpLabel
; CHECK:                  OpBranch %[[#bb154:]]
; CHECK:   %[[#bb154:]] = OpLabel
; CHECK:                  OpBranch %[[#bb140:]]
; CHECK:   %[[#bb140:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb158:]] None
; CHECK:                  OpSwitch %[[#]] %[[#bb158:]] 1 %[[#bb159:]] 2 %[[#bb160:]]
; CHECK:   %[[#bb159:]] = OpLabel
; CHECK:                  OpBranch %[[#bb158:]]
; CHECK:   %[[#bb160:]] = OpLabel
; CHECK:                  OpBranch %[[#bb158:]]
; CHECK:   %[[#bb158:]] = OpLabel
; CHECK:                  OpBranch %[[#bb137:]]
; CHECK:   %[[#bb137:]] = OpLabel
; CHECK:                  OpBranch %[[#bb134:]]
; CHECK:   %[[#bb134:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb161:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb162:]] %[[#bb161:]]
; CHECK:   %[[#bb162:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb163:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb164:]] %[[#bb165:]]
; CHECK:   %[[#bb164:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb166:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb167:]] %[[#bb168:]]
; CHECK:   %[[#bb165:]] = OpLabel
; CHECK:   %[[#bb167:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb169:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb170:]] %[[#bb171:]]
; CHECK:   %[[#bb168:]] = OpLabel
; CHECK:   %[[#bb170:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb172:]] None
; CHECK:                  OpSwitch %[[#]] %[[#bb173:]] 50 %[[#bb172:]] 51 %[[#bb174:]] 52 %[[#bb175:]] 53 %[[#bb176:]] 54 %[[#bb177:]]
; CHECK:   %[[#bb171:]] = OpLabel
; CHECK:   %[[#bb173:]] = OpLabel
; CHECK:                  OpBranch %[[#bb172:]]
; CHECK:   %[[#bb174:]] = OpLabel
; CHECK:                  OpBranch %[[#bb172:]]
; CHECK:   %[[#bb175:]] = OpLabel
; CHECK:                  OpBranch %[[#bb172:]]
; CHECK:   %[[#bb176:]] = OpLabel
; CHECK:                  OpBranch %[[#bb172:]]
; CHECK:   %[[#bb177:]] = OpLabel
; CHECK:                  OpBranch %[[#bb172:]]
; CHECK:   %[[#bb172:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb178:]] None
; CHECK:                  OpSwitch %[[#]] %[[#bb179:]] 1 %[[#bb178:]] 2 %[[#bb180:]] 3 %[[#bb181:]]
; CHECK:   %[[#bb179:]] = OpLabel
; CHECK:                  OpBranch %[[#bb178:]]
; CHECK:   %[[#bb180:]] = OpLabel
; CHECK:                  OpBranch %[[#bb178:]]
; CHECK:   %[[#bb181:]] = OpLabel
; CHECK:                  OpBranch %[[#bb178:]]
; CHECK:   %[[#bb178:]] = OpLabel
; CHECK:                  OpBranch %[[#bb169:]]
; CHECK:   %[[#bb169:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb182:]] None
; CHECK:                  OpSwitch %[[#]] %[[#bb183:]] 1 %[[#bb182:]] 2 %[[#bb184:]]
; CHECK:   %[[#bb183:]] = OpLabel
; CHECK:                  OpBranch %[[#bb182:]]
; CHECK:   %[[#bb184:]] = OpLabel
; CHECK:                  OpBranch %[[#bb182:]]
; CHECK:   %[[#bb182:]] = OpLabel
; CHECK:                  OpBranch %[[#bb166:]]
; CHECK:   %[[#bb166:]] = OpLabel
; CHECK:                  OpBranchConditional %[[#]] %[[#bb185:]] %[[#bb163:]]
; CHECK:   %[[#bb185:]] = OpLabel
; CHECK:                  OpBranch %[[#bb163:]]
; CHECK:   %[[#bb163:]] = OpLabel
; CHECK:                  OpBranch %[[#bb161:]]
; CHECK:   %[[#bb161:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb186:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb187:]] %[[#bb188:]]
; CHECK:   %[[#bb187:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb189:]] None
; CHECK:                  OpSwitch %[[#]] %[[#bb189:]] 35 %[[#bb190:]] 115 %[[#bb191:]]
; CHECK:   %[[#bb188:]] = OpLabel
; CHECK:   %[[#bb190:]] = OpLabel
; CHECK:                  OpBranch %[[#bb189:]]
; CHECK:   %[[#bb191:]] = OpLabel
; CHECK:                  OpBranch %[[#bb189:]]
; CHECK:   %[[#bb189:]] = OpLabel
; CHECK:                  OpBranchConditional %[[#]] %[[#bb186:]] %[[#bb192:]]
; CHECK:   %[[#bb192:]] = OpLabel
; CHECK:                  OpBranch %[[#bb186:]]
; CHECK:   %[[#bb186:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb193:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb194:]] %[[#bb193:]]
; CHECK:   %[[#bb194:]] = OpLabel
; CHECK:                  OpBranch %[[#bb193:]]
; CHECK:   %[[#bb193:]] = OpLabel
; CHECK:                  OpReturn
; CHECK:                  OpFunctionEnd
; CHECK: %[[#func_80:]] = OpFunction %[[#void:]] None %[[#]]
; CHECK:   %[[#bb195:]] = OpLabel
; CHECK:                  OpReturn
; CHECK:                  OpFunctionEnd



target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spirv-unknown-vulkan1.3-compute"

; Function Attrs: convergent noinline norecurse nounwind optnone
define spir_func noundef i32 @_Z3foov() #0 {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  ret i32 200
}

; Function Attrs: convergent nocallback nofree nosync nounwind willreturn memory(none)
declare token @llvm.experimental.convergence.entry() #1

; Function Attrs: convergent noinline norecurse nounwind optnone
define internal spir_func void @main() #0 {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  %result = alloca i32, align 4
  %a = alloca i32, align 4
  %c = alloca i32, align 4
  %r = alloca i32, align 4
  %s = alloca i32, align 4
  %t = alloca i32, align 4
  %sel = alloca float, align 4
  store i32 0, ptr %a, align 4
  %1 = load i32, ptr %a, align 4
  switch i32 %1, label %sw.default [
    i32 -3, label %sw.bb
    i32 0, label %sw.bb1
    i32 1, label %sw.bb2
    i32 2, label %sw.bb3
  ]

sw.bb:                                            ; preds = %entry
  store i32 -300, ptr %result, align 4
  br label %sw.epilog

sw.bb1:                                           ; preds = %entry
  store i32 0, ptr %result, align 4
  br label %sw.epilog

sw.bb2:                                           ; preds = %entry
  store i32 100, ptr %result, align 4
  br label %sw.epilog

sw.bb3:                                           ; preds = %entry
  %call4 = call spir_func noundef i32 @_Z3foov() #3 [ "convergencectrl"(token %0) ]
  store i32 %call4, ptr %result, align 4
  br label %sw.epilog

sw.default:                                       ; preds = %entry
  store i32 777, ptr %result, align 4
  br label %sw.epilog

sw.epilog:                                        ; preds = %sw.default, %sw.bb3, %sw.bb2, %sw.bb1, %sw.bb
  %2 = load i32, ptr %a, align 4
  store i32 %2, ptr %c, align 4
  %3 = load i32, ptr %c, align 4
  switch i32 %3, label %sw.epilog7 [
    i32 -4, label %sw.bb5
    i32 4, label %sw.bb6
  ]

sw.bb5:                                           ; preds = %sw.epilog
  store i32 -400, ptr %result, align 4
  br label %sw.epilog7

sw.bb6:                                           ; preds = %sw.epilog
  store i32 400, ptr %result, align 4
  br label %sw.epilog7

sw.epilog7:                                       ; preds = %sw.epilog, %sw.bb6, %sw.bb5
  %4 = load i32, ptr %a, align 4
  switch i32 %4, label %sw.epilog10 [
    i32 -5, label %sw.bb8
    i32 5, label %sw.bb9
  ]

sw.bb8:                                           ; preds = %sw.epilog7
  store i32 -500, ptr %result, align 4
  br label %sw.bb9

sw.bb9:                                           ; preds = %sw.epilog7, %sw.bb8
  store i32 500, ptr %result, align 4
  br label %sw.epilog10

sw.epilog10:                                      ; preds = %sw.bb9, %sw.epilog7
  %5 = load i32, ptr %a, align 4
  switch i32 %5, label %sw.default14 [
    i32 6, label %sw.bb11
    i32 7, label %sw.bb12
    i32 8, label %sw.bb13
  ]

sw.bb11:                                          ; preds = %sw.epilog10
  store i32 600, ptr %result, align 4
  br label %sw.bb12

sw.bb12:                                          ; preds = %sw.epilog10, %sw.bb11
  store i32 700, ptr %result, align 4
  br label %sw.bb13

sw.bb13:                                          ; preds = %sw.epilog10, %sw.bb12
  store i32 800, ptr %result, align 4
  br label %sw.epilog15

sw.default14:                                     ; preds = %sw.epilog10
  store i32 777, ptr %result, align 4
  br label %sw.epilog15

sw.epilog15:                                      ; preds = %sw.default14, %sw.bb13
  %6 = load i32, ptr %a, align 4
  switch i32 %6, label %sw.default17 [
    i32 10, label %sw.bb16
    i32 11, label %sw.bb16
    i32 12, label %sw.bb18
  ]

sw.bb16:                                          ; preds = %sw.epilog15, %sw.epilog15
  br label %sw.default17

sw.default17:                                     ; preds = %sw.epilog15, %sw.bb16
  br label %sw.bb18

sw.bb18:                                          ; preds = %sw.epilog15, %sw.default17
  store i32 12, ptr %result, align 4
  br label %sw.epilog19

sw.epilog19:                                      ; preds = %sw.bb18
  %7 = load i32, ptr %a, align 4
  switch i32 %7, label %sw.epilog21 [
    i32 15, label %sw.bb20
    i32 16, label %sw.bb20
  ]

sw.bb20:                                          ; preds = %sw.epilog19, %sw.epilog19
  br label %sw.epilog21

sw.epilog21:                                      ; preds = %sw.epilog19, %sw.bb20
  %8 = load i32, ptr %a, align 4
  switch i32 %8, label %sw.epilog29 [
    i32 20, label %sw.bb22
    i32 21, label %sw.bb23
    i32 22, label %sw.bb24
    i32 23, label %sw.bb24
    i32 24, label %sw.bb25
    i32 25, label %sw.bb25
    i32 26, label %sw.bb26
    i32 27, label %sw.bb26
    i32 28, label %sw.bb27
    i32 29, label %sw.bb28
  ]

sw.bb22:                                          ; preds = %sw.epilog21
  store i32 20, ptr %result, align 4
  br label %sw.epilog29

sw.bb23:                                          ; preds = %sw.epilog21
  store i32 21, ptr %result, align 4
  br label %sw.epilog29

sw.bb24:                                          ; preds = %sw.epilog21, %sw.epilog21
  br label %sw.epilog29

sw.bb25:                                          ; preds = %sw.epilog21, %sw.epilog21
  store i32 25, ptr %result, align 4
  br label %sw.epilog29

sw.bb26:                                          ; preds = %sw.epilog21, %sw.epilog21
  br label %sw.epilog29

sw.bb27:                                          ; preds = %sw.epilog21
  store i32 28, ptr %result, align 4
  br label %sw.epilog29

sw.bb28:                                          ; preds = %sw.epilog21
  store i32 29, ptr %result, align 4
  br label %sw.epilog29

sw.epilog29:                                      ; preds = %sw.epilog21, %sw.bb28, %sw.bb27, %sw.bb26, %sw.bb25, %sw.bb24, %sw.bb23, %sw.bb22
  %9 = load i32, ptr %a, align 4
  switch i32 %9, label %sw.epilog37 [
    i32 30, label %sw.bb30
  ]

sw.bb30:                                          ; preds = %sw.epilog29
  store i32 30, ptr %result, align 4
  %10 = load i32, ptr %result, align 4
  switch i32 %10, label %sw.default31 [
    i32 50, label %sw.bb32
    i32 51, label %sw.bb33
    i32 52, label %sw.bb33
    i32 53, label %sw.bb34
    i32 54, label %sw.bb35
  ]

sw.default31:                                     ; preds = %sw.bb30
  store i32 55, ptr %a, align 4
  br label %sw.bb32

sw.bb32:                                          ; preds = %sw.bb30, %sw.default31
  store i32 50, ptr %a, align 4
  br label %sw.epilog36

sw.bb33:                                          ; preds = %sw.bb30, %sw.bb30
  store i32 52, ptr %a, align 4
  br label %sw.bb34

sw.bb34:                                          ; preds = %sw.bb30, %sw.bb33
  store i32 53, ptr %a, align 4
  br label %sw.epilog36

sw.bb35:                                          ; preds = %sw.bb30
  store i32 54, ptr %a, align 4
  br label %sw.epilog36

sw.epilog36:                                      ; preds = %sw.bb35, %sw.bb34, %sw.bb32
  br label %sw.epilog37

sw.epilog37:                                      ; preds = %sw.epilog36, %sw.epilog29
  store i32 35, ptr %r, align 4
  store i32 45, ptr %s, align 4
  store i32 115, ptr %t, align 4
  %11 = load i32, ptr %a, align 4
  switch i32 %11, label %sw.epilog40 [
    i32 35, label %sw.bb38
    i32 115, label %sw.bb39
  ]

sw.bb38:                                          ; preds = %sw.epilog37
  store i32 35, ptr %result, align 4
  br label %sw.bb39

sw.bb39:                                          ; preds = %sw.epilog37, %sw.bb38
  store i32 115, ptr %result, align 4
  br label %sw.epilog40

sw.epilog40:                                      ; preds = %sw.epilog37, %sw.bb39
  %12 = load float, ptr %sel, align 4
  %conv = fptosi float %12 to i32
  switch i32 %conv, label %sw.epilog42 [
    i32 0, label %sw.bb41
  ]

sw.bb41:                                          ; preds = %sw.epilog40
  store i32 0, ptr %result, align 4
  br label %sw.epilog42

sw.epilog42:                                      ; preds = %sw.epilog40, %sw.bb41
  ret void
}

; Function Attrs: convergent norecurse
define void @main.1() #2 {
entry:
  call void @main()
  ret void
}

attributes #0 = { convergent noinline norecurse nounwind optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { convergent nocallback nofree nosync nounwind willreturn memory(none) }
attributes #2 = { convergent norecurse "frame-pointer"="all" "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #3 = { convergent }

!llvm.module.flags = !{!0, !1, !2}


!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 4, !"dx.disable_optimizations", i32 1}
!2 = !{i32 7, !"frame-pointer", i32 2}


