; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV: OpName %[[#r1:]] "r1"
; CHECK-SPIRV: OpName %[[#r2:]] "r2"
; CHECK-SPIRV: OpName %[[#r3:]] "r3"
; CHECK-SPIRV: OpName %[[#r4:]] "r4"
; CHECK-SPIRV: OpName %[[#r5:]] "r5"
; CHECK-SPIRV: OpName %[[#r6:]] "r6"
; CHECK-SPIRV: OpName %[[#r7:]] "r7"
; CHECK-SPIRV: OpName %[[#r8:]] "r8"
; CHECK-SPIRV: OpName %[[#r9:]] "r9"
; CHECK-SPIRV: OpName %[[#r10:]] "r10"
; CHECK-SPIRV: OpName %[[#r11:]] "r11"
; CHECK-SPIRV: OpName %[[#r12:]] "r12"
; CHECK-SPIRV: OpName %[[#r13:]] "r13"
; CHECK-SPIRV: OpName %[[#r14:]] "r14"
; CHECK-SPIRV: OpName %[[#r15:]] "r15"
; CHECK-SPIRV: OpName %[[#r16:]] "r16"
; CHECK-SPIRV: OpName %[[#r17:]] "r17"
; CHECK-SPIRV: OpName %[[#r18:]] "r18"
; CHECK-SPIRV: OpName %[[#r19:]] "r19"
; CHECK-SPIRV: OpName %[[#r20:]] "r20"
; CHECK-SPIRV: OpName %[[#r21:]] "r21"
; CHECK-SPIRV: OpName %[[#r22:]] "r22"
; CHECK-SPIRV: OpName %[[#r23:]] "r23"
; CHECK-SPIRV: OpName %[[#r24:]] "r24"
; CHECK-SPIRV: OpName %[[#r25:]] "r25"
; CHECK-SPIRV: OpName %[[#r26:]] "r26"
; CHECK-SPIRV: OpName %[[#r27:]] "r27"
; CHECK-SPIRV: OpName %[[#r28:]] "r28"
; CHECK-SPIRV: OpName %[[#r29:]] "r29"
; CHECK-SPIRV: OpName %[[#r30:]] "r30"
; CHECK-SPIRV: OpName %[[#r31:]] "r31"
; CHECK-SPIRV: OpName %[[#r32:]] "r32"
; CHECK-SPIRV: OpName %[[#r33:]] "r33"
; CHECK-SPIRV: OpName %[[#r34:]] "r34"
; CHECK-SPIRV: OpName %[[#r35:]] "r35"
; CHECK-SPIRV: OpName %[[#r36:]] "r36"
; CHECK-SPIRV: OpName %[[#r37:]] "r37"
; CHECK-SPIRV: OpName %[[#r38:]] "r38"
; CHECK-SPIRV: OpName %[[#r39:]] "r39"
; CHECK-SPIRV: OpName %[[#r40:]] "r40"
; CHECK-SPIRV: OpName %[[#r41:]] "r41"
; CHECK-SPIRV: OpName %[[#r42:]] "r42"
; CHECK-SPIRV: OpName %[[#r43:]] "r43"
; CHECK-SPIRV: OpName %[[#r44:]] "r44"
; CHECK-SPIRV: OpName %[[#r45:]] "r45"
; CHECK-SPIRV: OpName %[[#r46:]] "r46"
; CHECK-SPIRV: OpName %[[#r47:]] "r47"
; CHECK-SPIRV: OpName %[[#r48:]] "r48"
; CHECK-SPIRV: OpName %[[#r49:]] "r49"
; CHECK-SPIRV: OpName %[[#r50:]] "r50"
; CHECK-SPIRV: OpName %[[#r51:]] "r51"
; CHECK-SPIRV: OpName %[[#r52:]] "r52"
; CHECK-SPIRV: OpName %[[#r53:]] "r53"
; CHECK-SPIRV: OpName %[[#r54:]] "r54"
; CHECK-SPIRV: OpName %[[#r55:]] "r55"
; CHECK-SPIRV: OpName %[[#r56:]] "r56"
; CHECK-SPIRV: OpName %[[#r57:]] "r57"
; CHECK-SPIRV: OpName %[[#r58:]] "r58"
; CHECK-SPIRV: OpName %[[#r59:]] "r59"
; CHECK-SPIRV: OpName %[[#r60:]] "r60"
; CHECK-SPIRV: OpName %[[#r61:]] "r61"
; CHECK-SPIRV: OpName %[[#r62:]] "r62"
; CHECK-SPIRV: OpName %[[#r63:]] "r63"
; CHECK-SPIRV: OpName %[[#r64:]] "r64"
; CHECK-SPIRV: OpName %[[#r65:]] "r65"
; CHECK-SPIRV: OpName %[[#r66:]] "r66"
; CHECK-SPIRV: OpName %[[#r67:]] "r67"
; CHECK-SPIRV: OpName %[[#r68:]] "r68"
; CHECK-SPIRV: OpName %[[#r69:]] "r69"
; CHECK-SPIRV: OpName %[[#r70:]] "r70"
; CHECK-SPIRV: OpName %[[#r71:]] "r71"
; CHECK-SPIRV: OpName %[[#r72:]] "r72"
; CHECK-SPIRV: OpName %[[#r73:]] "r73"
; CHECK-SPIRV: OpName %[[#r74:]] "r74"
; CHECK-SPIRV: OpName %[[#r75:]] "r75"
; CHECK-SPIRV: OpName %[[#r76:]] "r76"
; CHECK-SPIRV: OpName %[[#r77:]] "r77"
; CHECK-SPIRV: OpName %[[#r78:]] "r78"
; CHECK-SPIRV: OpName %[[#r79:]] "r79"
; CHECK-SPIRV: OpName %[[#r80:]] "r80"
; CHECK-SPIRV: OpName %[[#r81:]] "r81"
; CHECK-SPIRV: OpName %[[#r82:]] "r82"
; CHECK-SPIRV: OpName %[[#r83:]] "r83"
; CHECK-SPIRV: OpName %[[#r84:]] "r84"
; CHECK-SPIRV: OpName %[[#r85:]] "r85"
; CHECK-SPIRV: OpName %[[#r86:]] "r86"
; CHECK-SPIRV: OpName %[[#r87:]] "r87"
; CHECK-SPIRV: OpName %[[#r88:]] "r88"
; CHECK-SPIRV: OpName %[[#r89:]] "r89"
; CHECK-SPIRV: OpName %[[#r90:]] "r90"
; CHECK-SPIRV-NOT: OpDecorate %{{.*}} FPFastMathMode
; CHECK-SPIRV: %[[#bool:]] = OpTypeBool
; CHECK-SPIRV: %[[#r1]] = OpFOrdEqual %[[#bool]]
; CHECK-SPIRV: %[[#r2]] = OpFOrdEqual %[[#bool]]
; CHECK-SPIRV: %[[#r3]] = OpFOrdEqual %[[#bool]]
; CHECK-SPIRV: %[[#r4]] = OpFOrdEqual %[[#bool]]
; CHECK-SPIRV: %[[#r5]] = OpFOrdEqual %[[#bool]]
; CHECK-SPIRV: %[[#r6]] = OpFOrdEqual %[[#bool]]
; CHECK-SPIRV: %[[#r7]] = OpFOrdEqual %[[#bool]]
; CHECK-SPIRV: %[[#r8]] = OpFOrdNotEqual %[[#bool]]
; CHECK-SPIRV: %[[#r9]] = OpFOrdNotEqual %[[#bool]]
; CHECK-SPIRV: %[[#r10]] = OpFOrdNotEqual %[[#bool]]
; CHECK-SPIRV: %[[#r11]] = OpFOrdNotEqual %[[#bool]]
; CHECK-SPIRV: %[[#r12]] = OpFOrdNotEqual %[[#bool]]
; CHECK-SPIRV: %[[#r13]] = OpFOrdNotEqual %[[#bool]]
; CHECK-SPIRV: %[[#r14]] = OpFOrdNotEqual %[[#bool]]
; CHECK-SPIRV: %[[#r15]] = OpFOrdLessThan %[[#bool]]
; CHECK-SPIRV: %[[#r16]] = OpFOrdLessThan %[[#bool]]
; CHECK-SPIRV: %[[#r17]] = OpFOrdLessThan %[[#bool]]
; CHECK-SPIRV: %[[#r18]] = OpFOrdLessThan %[[#bool]]
; CHECK-SPIRV: %[[#r19]] = OpFOrdLessThan %[[#bool]]
; CHECK-SPIRV: %[[#r20]] = OpFOrdLessThan %[[#bool]]
; CHECK-SPIRV: %[[#r21]] = OpFOrdLessThan %[[#bool]]
; CHECK-SPIRV: %[[#r22]] = OpFOrdGreaterThan %[[#bool]]
; CHECK-SPIRV: %[[#r23]] = OpFOrdGreaterThan %[[#bool]]
; CHECK-SPIRV: %[[#r24]] = OpFOrdGreaterThan %[[#bool]]
; CHECK-SPIRV: %[[#r25]] = OpFOrdGreaterThan %[[#bool]]
; CHECK-SPIRV: %[[#r26]] = OpFOrdGreaterThan %[[#bool]]
; CHECK-SPIRV: %[[#r27]] = OpFOrdGreaterThan %[[#bool]]
; CHECK-SPIRV: %[[#r28]] = OpFOrdGreaterThan %[[#bool]]
; CHECK-SPIRV: %[[#r29]] = OpFOrdLessThanEqual %[[#bool]]
; CHECK-SPIRV: %[[#r30]] = OpFOrdLessThanEqual %[[#bool]]
; CHECK-SPIRV: %[[#r31]] = OpFOrdLessThanEqual %[[#bool]]
; CHECK-SPIRV: %[[#r32]] = OpFOrdLessThanEqual %[[#bool]]
; CHECK-SPIRV: %[[#r33]] = OpFOrdLessThanEqual %[[#bool]]
; CHECK-SPIRV: %[[#r34]] = OpFOrdLessThanEqual %[[#bool]]
; CHECK-SPIRV: %[[#r35]] = OpFOrdLessThanEqual %[[#bool]]
; CHECK-SPIRV: %[[#r36]] = OpFOrdGreaterThanEqual %[[#bool]]
; CHECK-SPIRV: %[[#r37]] = OpFOrdGreaterThanEqual %[[#bool]]
; CHECK-SPIRV: %[[#r38]] = OpFOrdGreaterThanEqual %[[#bool]]
; CHECK-SPIRV: %[[#r39]] = OpFOrdGreaterThanEqual %[[#bool]]
; CHECK-SPIRV: %[[#r40]] = OpFOrdGreaterThanEqual %[[#bool]]
; CHECK-SPIRV: %[[#r41]] = OpFOrdGreaterThanEqual %[[#bool]]
; CHECK-SPIRV: %[[#r42]] = OpFOrdGreaterThanEqual %[[#bool]]
; CHECK-SPIRV: %[[#r43]] = OpOrdered %[[#bool]]
; CHECK-SPIRV: %[[#r44]] = OpOrdered %[[#bool]]
; CHECK-SPIRV: %[[#r45]] = OpOrdered %[[#bool]]
; CHECK-SPIRV: %[[#r46]] = OpFUnordEqual %[[#bool]]
; CHECK-SPIRV: %[[#r47]] = OpFUnordEqual %[[#bool]]
; CHECK-SPIRV: %[[#r48]] = OpFUnordEqual %[[#bool]]
; CHECK-SPIRV: %[[#r49]] = OpFUnordEqual %[[#bool]]
; CHECK-SPIRV: %[[#r50]] = OpFUnordEqual %[[#bool]]
; CHECK-SPIRV: %[[#r51]] = OpFUnordEqual %[[#bool]]
; CHECK-SPIRV: %[[#r52]] = OpFUnordEqual %[[#bool]]
; CHECK-SPIRV: %[[#r53]] = OpFUnordNotEqual %[[#bool]]
; CHECK-SPIRV: %[[#r54]] = OpFUnordNotEqual %[[#bool]]
; CHECK-SPIRV: %[[#r55]] = OpFUnordNotEqual %[[#bool]]
; CHECK-SPIRV: %[[#r56]] = OpFUnordNotEqual %[[#bool]]
; CHECK-SPIRV: %[[#r57]] = OpFUnordNotEqual %[[#bool]]
; CHECK-SPIRV: %[[#r58]] = OpFUnordNotEqual %[[#bool]]
; CHECK-SPIRV: %[[#r59]] = OpFUnordNotEqual %[[#bool]]
; CHECK-SPIRV: %[[#r60]] = OpFUnordLessThan %[[#bool]]
; CHECK-SPIRV: %[[#r61]] = OpFUnordLessThan %[[#bool]]
; CHECK-SPIRV: %[[#r62]] = OpFUnordLessThan %[[#bool]]
; CHECK-SPIRV: %[[#r63]] = OpFUnordLessThan %[[#bool]]
; CHECK-SPIRV: %[[#r64]] = OpFUnordLessThan %[[#bool]]
; CHECK-SPIRV: %[[#r65]] = OpFUnordLessThan %[[#bool]]
; CHECK-SPIRV: %[[#r66]] = OpFUnordLessThan %[[#bool]]
; CHECK-SPIRV: %[[#r67]] = OpFUnordGreaterThan %[[#bool]]
; CHECK-SPIRV: %[[#r68]] = OpFUnordGreaterThan %[[#bool]]
; CHECK-SPIRV: %[[#r69]] = OpFUnordGreaterThan %[[#bool]]
; CHECK-SPIRV: %[[#r70]] = OpFUnordGreaterThan %[[#bool]]
; CHECK-SPIRV: %[[#r71]] = OpFUnordGreaterThan %[[#bool]]
; CHECK-SPIRV: %[[#r72]] = OpFUnordGreaterThan %[[#bool]]
; CHECK-SPIRV: %[[#r73]] = OpFUnordGreaterThan %[[#bool]]
; CHECK-SPIRV: %[[#r74]] = OpFUnordLessThanEqual %[[#bool]]
; CHECK-SPIRV: %[[#r75]] = OpFUnordLessThanEqual %[[#bool]]
; CHECK-SPIRV: %[[#r76]] = OpFUnordLessThanEqual %[[#bool]]
; CHECK-SPIRV: %[[#r77]] = OpFUnordLessThanEqual %[[#bool]]
; CHECK-SPIRV: %[[#r78]] = OpFUnordLessThanEqual %[[#bool]]
; CHECK-SPIRV: %[[#r79]] = OpFUnordLessThanEqual %[[#bool]]
; CHECK-SPIRV: %[[#r80]] = OpFUnordLessThanEqual %[[#bool]]
; CHECK-SPIRV: %[[#r81]] = OpFUnordGreaterThanEqual %[[#bool]]
; CHECK-SPIRV: %[[#r82]] = OpFUnordGreaterThanEqual %[[#bool]]
; CHECK-SPIRV: %[[#r83]] = OpFUnordGreaterThanEqual %[[#bool]]
; CHECK-SPIRV: %[[#r84]] = OpFUnordGreaterThanEqual %[[#bool]]
; CHECK-SPIRV: %[[#r85]] = OpFUnordGreaterThanEqual %[[#bool]]
; CHECK-SPIRV: %[[#r86]] = OpFUnordGreaterThanEqual %[[#bool]]
; CHECK-SPIRV: %[[#r87]] = OpFUnordGreaterThanEqual %[[#bool]]
; CHECK-SPIRV: %[[#r88]] = OpUnordered %[[#bool]]
; CHECK-SPIRV: %[[#r89]] = OpUnordered %[[#bool]]
; CHECK-SPIRV: %[[#r90]] = OpUnordered %[[#bool]]

@G = global [90 x i1] zeroinitializer

define spir_kernel void @testFCmp(float %a, float %b) local_unnamed_addr {
entry:
  %r1 = fcmp oeq float %a, %b
  %r2 = fcmp nnan oeq float %a, %b
  %r3 = fcmp ninf oeq float %a, %b
  %r4 = fcmp nsz oeq float %a, %b
  %r5 = fcmp arcp oeq float %a, %b
  %r6 = fcmp fast oeq float %a, %b
  %r7 = fcmp nnan ninf oeq float %a, %b
  %r8 = fcmp one float %a, %b
  %r9 = fcmp nnan one float %a, %b
  %r10 = fcmp ninf one float %a, %b
  %r11 = fcmp nsz one float %a, %b
  %r12 = fcmp arcp one float %a, %b
  %r13 = fcmp fast one float %a, %b
  %r14 = fcmp nnan ninf one float %a, %b
  %r15 = fcmp olt float %a, %b
  %r16 = fcmp nnan olt float %a, %b
  %r17 = fcmp ninf olt float %a, %b
  %r18 = fcmp nsz olt float %a, %b
  %r19 = fcmp arcp olt float %a, %b
  %r20 = fcmp fast olt float %a, %b
  %r21 = fcmp nnan ninf olt float %a, %b
  %r22 = fcmp ogt float %a, %b
  %r23 = fcmp nnan ogt float %a, %b
  %r24 = fcmp ninf ogt float %a, %b
  %r25 = fcmp nsz ogt float %a, %b
  %r26 = fcmp arcp ogt float %a, %b
  %r27 = fcmp fast ogt float %a, %b
  %r28 = fcmp nnan ninf ogt float %a, %b
  %r29 = fcmp ole float %a, %b
  %r30 = fcmp nnan ole float %a, %b
  %r31 = fcmp ninf ole float %a, %b
  %r32 = fcmp nsz ole float %a, %b
  %r33 = fcmp arcp ole float %a, %b
  %r34 = fcmp fast ole float %a, %b
  %r35 = fcmp nnan ninf ole float %a, %b
  %r36 = fcmp oge float %a, %b
  %r37 = fcmp nnan oge float %a, %b
  %r38 = fcmp ninf oge float %a, %b
  %r39 = fcmp nsz oge float %a, %b
  %r40 = fcmp arcp oge float %a, %b
  %r41 = fcmp fast oge float %a, %b
  %r42 = fcmp nnan ninf oge float %a, %b
  %r43 = fcmp ord float %a, %b
  %r44 = fcmp ninf ord float %a, %b
  %r45 = fcmp nsz ord float %a, %b
  %r46 = fcmp ueq float %a, %b
  %r47 = fcmp nnan ueq float %a, %b
  %r48 = fcmp ninf ueq float %a, %b
  %r49 = fcmp nsz ueq float %a, %b
  %r50 = fcmp arcp ueq float %a, %b
  %r51 = fcmp fast ueq float %a, %b
  %r52 = fcmp nnan ninf ueq float %a, %b
  %r53 = fcmp une float %a, %b
  %r54 = fcmp nnan une float %a, %b
  %r55 = fcmp ninf une float %a, %b
  %r56 = fcmp nsz une float %a, %b
  %r57 = fcmp arcp une float %a, %b
  %r58 = fcmp fast une float %a, %b
  %r59 = fcmp nnan ninf une float %a, %b
  %r60 = fcmp ult float %a, %b
  %r61 = fcmp nnan ult float %a, %b
  %r62 = fcmp ninf ult float %a, %b
  %r63 = fcmp nsz ult float %a, %b
  %r64 = fcmp arcp ult float %a, %b
  %r65 = fcmp fast ult float %a, %b
  %r66 = fcmp nnan ninf ult float %a, %b
  %r67 = fcmp ugt float %a, %b
  %r68 = fcmp nnan ugt float %a, %b
  %r69 = fcmp ninf ugt float %a, %b
  %r70 = fcmp nsz ugt float %a, %b
  %r71 = fcmp arcp ugt float %a, %b
  %r72 = fcmp fast ugt float %a, %b
  %r73 = fcmp nnan ninf ugt float %a, %b
  %r74 = fcmp ule float %a, %b
  %r75 = fcmp nnan ule float %a, %b
  %r76 = fcmp ninf ule float %a, %b
  %r77 = fcmp nsz ule float %a, %b
  %r78 = fcmp arcp ule float %a, %b
  %r79 = fcmp fast ule float %a, %b
  %r80 = fcmp nnan ninf ule float %a, %b
  %r81 = fcmp uge float %a, %b
  %r82 = fcmp nnan uge float %a, %b
  %r83 = fcmp ninf uge float %a, %b
  %r84 = fcmp nsz uge float %a, %b
  %r85 = fcmp arcp uge float %a, %b
  %r86 = fcmp fast uge float %a, %b
  %r87 = fcmp nnan ninf uge float %a, %b
  %r88 = fcmp uno float %a, %b
  %r89 = fcmp ninf uno float %a, %b
  %r90 = fcmp nsz uno float %a, %b
  %p1 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 0
  store i1 %r1, ptr %p1
  %p2 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 1
  store i1 %r2, ptr %p2
  %p3 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 2
  store i1 %r3, ptr %p3
  %p4 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 3
  store i1 %r4, ptr %p4
  %p5 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 4
  store i1 %r5, ptr %p5
  %p6 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 5
  store i1 %r6, ptr %p6
  %p7 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 6
  store i1 %r7, ptr %p7
  %p8 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 7
  store i1 %r8, ptr %p8
  %p9 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 8
  store i1 %r9, ptr %p9
  %p10 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 9
  store i1 %r10, ptr %p10
  %p11 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 10
  store i1 %r11, ptr %p11
  %p12 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 11
  store i1 %r12, ptr %p12
  %p13 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 12
  store i1 %r13, ptr %p13
  %p14 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 13
  store i1 %r14, ptr %p14
  %p15 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 14
  store i1 %r15, ptr %p15
  %p16 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 15
  store i1 %r16, ptr %p16
  %p17 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 16
  store i1 %r17, ptr %p17
  %p18 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 17
  store i1 %r18, ptr %p18
  %p19 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 18
  store i1 %r19, ptr %p19
  %p20 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 19
  store i1 %r20, ptr %p20
  %p21 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 20
  store i1 %r21, ptr %p21
  %p22 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 21
  store i1 %r22, ptr %p22
  %p23 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 22
  store i1 %r23, ptr %p23
  %p24 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 23
  store i1 %r24, ptr %p24
  %p25 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 24
  store i1 %r25, ptr %p25
  %p26 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 25
  store i1 %r26, ptr %p26
  %p27 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 26
  store i1 %r27, ptr %p27
  %p28 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 27
  store i1 %r28, ptr %p28
  %p29 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 28
  store i1 %r29, ptr %p29
  %p30 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 29
  store i1 %r30, ptr %p30
  %p31 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 30
  store i1 %r31, ptr %p31
  %p32 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 31
  store i1 %r32, ptr %p32
  %p33 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 32
  store i1 %r33, ptr %p33
  %p34 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 33
  store i1 %r34, ptr %p34
  %p35 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 34
  store i1 %r35, ptr %p35
  %p36 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 35
  store i1 %r36, ptr %p36
  %p37 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 36
  store i1 %r37, ptr %p37
  %p38 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 37
  store i1 %r38, ptr %p38
  %p39 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 38
  store i1 %r39, ptr %p39
  %p40 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 39
  store i1 %r40, ptr %p40
  %p41 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 40
  store i1 %r41, ptr %p41
  %p42 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 41
  store i1 %r42, ptr %p42
  %p43 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 42
  store i1 %r43, ptr %p43
  %p44 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 43
  store i1 %r44, ptr %p44
  %p45 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 44
  store i1 %r45, ptr %p45
  %p46 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 45
  store i1 %r46, ptr %p46
  %p47 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 46
  store i1 %r47, ptr %p47
  %p48 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 47
  store i1 %r48, ptr %p48
  %p49 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 48
  store i1 %r49, ptr %p49
  %p50 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 49
  store i1 %r50, ptr %p50
  %p51 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 50
  store i1 %r51, ptr %p51
  %p52 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 51
  store i1 %r52, ptr %p52
  %p53 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 52
  store i1 %r53, ptr %p53
  %p54 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 53
  store i1 %r54, ptr %p54
  %p55 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 54
  store i1 %r55, ptr %p55
  %p56 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 55
  store i1 %r56, ptr %p56
  %p57 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 56
  store i1 %r57, ptr %p57
  %p58 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 57
  store i1 %r58, ptr %p58
  %p59 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 58
  store i1 %r59, ptr %p59
  %p60 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 59
  store i1 %r60, ptr %p60
  %p61 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 60
  store i1 %r61, ptr %p61
  %p62 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 61
  store i1 %r62, ptr %p62
  %p63 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 62
  store i1 %r63, ptr %p63
  %p64 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 63
  store i1 %r64, ptr %p64
  %p65 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 64
  store i1 %r65, ptr %p65
  %p66 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 65
  store i1 %r66, ptr %p66
  %p67 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 66
  store i1 %r67, ptr %p67
  %p68 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 67
  store i1 %r68, ptr %p68
  %p69 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 68
  store i1 %r69, ptr %p69
  %p70 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 69
  store i1 %r70, ptr %p70
  %p71 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 70
  store i1 %r71, ptr %p71
  %p72 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 71
  store i1 %r72, ptr %p72
  %p73 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 72
  store i1 %r73, ptr %p73
  %p74 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 73
  store i1 %r74, ptr %p74
  %p75 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 74
  store i1 %r75, ptr %p75
  %p76 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 75
  store i1 %r76, ptr %p76
  %p77 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 76
  store i1 %r77, ptr %p77
  %p78 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 77
  store i1 %r78, ptr %p78
  %p79 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 78
  store i1 %r79, ptr %p79
  %p80 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 79
  store i1 %r80, ptr %p80
  %p81 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 80
  store i1 %r81, ptr %p81
  %p82 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 81
  store i1 %r82, ptr %p82
  %p83 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 82
  store i1 %r83, ptr %p83
  %p84 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 83
  store i1 %r84, ptr %p84
  %p85 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 84
  store i1 %r85, ptr %p85
  %p86 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 85
  store i1 %r86, ptr %p86
  %p87 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 86
  store i1 %r87, ptr %p87
  %p88 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 87
  store i1 %r88, ptr %p88
  %p89 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 88
  store i1 %r89, ptr %p89
  %p90 = getelementptr inbounds [90 x i1], ptr @G, i32 0, i32 89
  store i1 %r90, ptr %p90
  ret void
}
