; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

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

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spirv32-unknown-unknown"

; Function Attrs: nounwind
define spir_kernel void @testFCmp(float %a, float %b) local_unnamed_addr #0 !kernel_arg_addr_space !2 !kernel_arg_access_qual !3 !kernel_arg_type !4 !kernel_arg_base_type !4 !kernel_arg_type_qual !5 {
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
  ret void
}

attributes #0 = { convergent nounwind writeonly "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "uniform-work-group-size"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 2, i32 0}
!2 = !{i32 0, i32 0}
!3 = !{!"none", !"none"}
!4 = !{!"float", !"float"}
!5 = !{!"", !""}
