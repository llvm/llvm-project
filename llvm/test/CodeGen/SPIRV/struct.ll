; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s

%struct.ST = type { i32, i32, i32 }

; CHECK-DAG: OpName %[[#struct:]] "struct.ST"
; CHECK-DAG: %[[#int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#struct]] = OpTypeStruct %[[#int]] %[[#int]] %[[#int]]
; CHECK-DAG: %[[#structP:]] = OpTypePointer Function %[[#struct]]
; CHECK-DAG: %[[#intP:]] = OpTypePointer Function %[[#int]]
; CHECK-DAG: %[[#zero:]] = OpConstant %[[#int]] 0
; CHECK-DAG: %[[#one:]] = OpConstant %[[#int]] 1
; CHECK-DAG: %[[#two:]] = OpConstant %[[#int]] 2
; CHECK-DAG: %[[#three:]] = OpConstant %[[#int]] 3

define dso_local spir_func i32 @func() {
entry:
; CHECK-SPIRV: %[[#st:]] = OpVariable %[[#structP]]
  %st = alloca %struct.ST, align 4
; CHECK-SPIRV: %[[#a:]] = OpInBoundsPtrAccessChain %[[#intP]] %[[#st]] %[[#zero]] %[[#zero]]
; CHECK-SPIRV: OpStore %[[#a]] %[[#one]]
  %a = getelementptr inbounds %struct.ST, %struct.ST* %st, i32 0, i32 0
  store i32 1, i32* %a, align 4
; CHECK-SPIRV: %[[#b:]] = OpInBoundsPtrAccessChain %[[#intP]] %[[#st]] %[[#zero]] %[[#one]]
; CHECK-SPIRV: OpStore %[[#b]] %[[#two]]
  %b = getelementptr inbounds %struct.ST, %struct.ST* %st, i32 0, i32 1
  store i32 2, i32* %b, align 4
; CHECK-SPIRV: %[[#c:]] = OpInBoundsPtrAccessChain %[[#intP]] %[[#st]] %[[#zero]] %[[#two]]
; CHECK-SPIRV: OpStore %[[#c]] %[[#three]]
  %c = getelementptr inbounds %struct.ST, %struct.ST* %st, i32 0, i32 2
  store i32 3, i32* %c, align 4
; CHECK-SPIRV: %[[#a1:]] = OpInBoundsPtrAccessChain %[[#intP]] %[[#st]] %[[#zero]] %[[#zero]]
; CHECK-SPIRV: %[[#]] = OpLoad %[[#int]] %[[#a1]]
  %a1 = getelementptr inbounds %struct.ST, %struct.ST* %st, i32 0, i32 0
  %0 = load i32, i32* %a1, align 4
; CHECK-SPIRV: %[[#b1:]] = OpInBoundsPtrAccessChain %[[#intP]] %[[#st]] %[[#zero]] %[[#one]]
; CHECK-SPIRV: %[[#]] = OpLoad %[[#int]] %[[#b1]]
  %b2 = getelementptr inbounds %struct.ST, %struct.ST* %st, i32 0, i32 1
  %1 = load i32, i32* %b2, align 4
  %add = add nsw i32 %0, %1
; CHECK-SPIRV: %[[#c1:]] = OpInBoundsPtrAccessChain %[[#intP]] %[[#st]] %[[#zero]] %[[#two]]
; CHECK-SPIRV: %[[#]] = OpLoad %[[#int]] %[[#c1]]
  %c3 = getelementptr inbounds %struct.ST, %struct.ST* %st, i32 0, i32 2
  %2 = load i32, i32* %c3, align 4
  %add4 = add nsw i32 %add, %2
  ret i32 %add4
}
