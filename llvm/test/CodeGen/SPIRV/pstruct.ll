; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

%struct.ST = type { i32, i32, i32 }

; CHECK-SPIRV-DAG: OpName %[[#struct:]] "struct.ST"
; CHECK-SPIRV-DAG: %[[#int:]] = OpTypeInt 32 0
; CHECK-SPIRV-DAG: %[[#intP:]] = OpTypePointer Function %[[#int]]
; CHECK-SPIRV-DAG: %[[#struct]] = OpTypeStruct %[[#int]] %[[#int]] %[[#int]]
; CHECK-SPIRV-DAG: %[[#structP:]] = OpTypePointer Function %[[#struct]]
; CHECK-SPIRV-DAG: %[[#structPP:]] = OpTypePointer Function %[[#structP]]
; CHECK-SPIRV-DAG: %[[#zero:]] = OpConstantNull %[[#int]]
; CHECK-SPIRV-DAG: %[[#one:]] = OpConstant %[[#int]] 1{{$}}
; CHECK-SPIRV-DAG: %[[#two:]] = OpConstant %[[#int]] 2{{$}}

define dso_local spir_func i32 @cmp_func(ptr %p1, ptr %p2) {
entry:
; CHECK-SPIRV: %[[#retval:]] = OpVariable %[[#intP]] Function
; CHECK-SPIRV: %[[#p1_addr:]] = OpVariable %[[#ptrPtrTy:]] Function
; CHECK-SPIRV: %[[#p2_addr:]] = OpVariable %[[#ptrPtrTy]] Function
; CHECK-SPIRV: %[[#s1:]] = OpVariable %[[#ptrPtrTy]] Function
; CHECK-SPIRV: %[[#s2:]] = OpVariable %[[#ptrPtrTy]] Function
  %retval = alloca i32, align 4
  %p1.addr = alloca ptr, align 8
  %p2.addr = alloca ptr, align 8
  %s1 = alloca ptr, align 8
  %s2 = alloca ptr, align 8
; CHECK-SPIRV: OpStore %[[#p1_addr]] %[[#p1:]] Aligned 8
  store ptr %p1, ptr %p1.addr, align 8
; CHECK-SPIRV: OpStore %[[#p2_addr]] %[[#p2:]] Aligned 8
  store ptr %p2, ptr %p2.addr, align 8
; CHECK-SPIRV: %[[#val0:]] = OpLoad %[[#ptrTy:]] %[[#p1_addr]] Aligned 8
  %0 = load ptr, ptr %p1.addr, align 8
; CHECK-SPIRV: OpStore %[[#s1]] %[[#val0]] Aligned 8
  store ptr %0, ptr %s1, align 8
; CHECK-SPIRV: %[[#val1:]] = OpLoad %[[#ptrTy]] %[[#p2_addr]] Aligned 8
  %2 = load ptr, ptr %p2.addr, align 8
; CHECK-SPIRV: OpStore %[[#s2]] %[[#val1]] Aligned 8
  store ptr %2, ptr %s2, align 8
; CHECK-SPIRV: %[[#]] = OpBitcast %[[#structPP]]
; CHECK-SPIRV: %[[#t3:]] = OpLoad %[[#structP]]
; CHECK-SPIRV: %[[#a1:]] = OpInBoundsPtrAccessChain %[[#intP]] %[[#t3]] %[[#zero]] %[[#zero]]
; CHECK-SPIRV: %[[#]] = OpLoad %[[#int]] %[[#a1]]
  %4 = load ptr, ptr %s1, align 8
  %a = getelementptr inbounds %struct.ST, ptr %4, i32 0, i32 0
  %5 = load i32, ptr %a, align 4
; CHECK-SPIRV: %[[#]] = OpBitcast %[[#structPP]]
; CHECK-SPIRV: %[[#t4:]] = OpLoad %[[#structP]]
; CHECK-SPIRV: %[[#a2:]] = OpInBoundsPtrAccessChain %[[#intP]] %[[#t4]] %[[#zero]] %[[#zero]]
; CHECK-SPIRV: %[[#]] = OpLoad %[[#int]] %[[#a2]]
  %6 = load ptr, ptr %s2, align 8
  %a1 = getelementptr inbounds %struct.ST, ptr %6, i32 0, i32 0
  %7 = load i32, ptr %a1, align 4
  %cmp = icmp ne i32 %5, %7
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %8 = load ptr, ptr %s1, align 8
  %a2 = getelementptr inbounds %struct.ST, ptr %8, i32 0, i32 0
  %9 = load i32, ptr %a2, align 4
  %10 = load ptr, ptr %s2, align 8
  %a3 = getelementptr inbounds %struct.ST, ptr %10, i32 0, i32 0
  %11 = load i32, ptr %a3, align 4
  %sub = sub nsw i32 %9, %11
  store i32 %sub, ptr %retval, align 4
  br label %return

if.end:                                           ; preds = %entry
; CHECK-SPIRV: %[[#]] = OpBitcast %[[#structPP]]
; CHECK-SPIRV: %[[#t7:]] = OpLoad %[[#structP]]
; CHECK-SPIRV: %[[#b1:]] = OpInBoundsPtrAccessChain %[[#intP]] %[[#t7]] %[[#zero]] %[[#one]]
; CHECK-SPIRV: %[[#]] = OpLoad %[[#int]] %[[#b1]]
  %12 = load ptr, ptr %s1, align 8
  %b = getelementptr inbounds %struct.ST, ptr %12, i32 0, i32 1
  %13 = load i32, ptr %b, align 4
; CHECK-SPIRV: %[[#]] = OpBitcast %[[#structPP]]
; CHECK-SPIRV: %[[#t8:]] = OpLoad %[[#structP]]
; CHECK-SPIRV: %[[#b2:]] = OpInBoundsPtrAccessChain %[[#intP]] %[[#t8]] %[[#zero]] %[[#one]]
; CHECK-SPIRV: %[[#]] = OpLoad %[[#int]] %[[#b2]]
  %14 = load ptr, ptr %s2, align 8
  %b4 = getelementptr inbounds %struct.ST, ptr %14, i32 0, i32 1
  %15 = load i32, ptr %b4, align 4
  %cmp5 = icmp ne i32 %13, %15
  br i1 %cmp5, label %if.then6, label %if.end10

if.then6:                                         ; preds = %if.end
  %16 = load ptr, ptr %s1, align 8
  %b7 = getelementptr inbounds %struct.ST, ptr %16, i32 0, i32 1
  %17 = load i32, ptr %b7, align 4
  %18 = load ptr, ptr %s2, align 8
  %b8 = getelementptr inbounds %struct.ST, ptr %18, i32 0, i32 1
  %19 = load i32, ptr %b8, align 4
  %sub9 = sub nsw i32 %17, %19
  store i32 %sub9, ptr %retval, align 4
  br label %return

if.end10:                                         ; preds = %if.end
; CHECK-SPIRV: %[[#]] = OpBitcast %[[#structPP]]
; CHECK-SPIRV: %[[#t11:]] = OpLoad %[[#structP]]
; CHECK-SPIRV: %[[#c1:]] = OpInBoundsPtrAccessChain %[[#intP]] %[[#t11]] %[[#zero]] %[[#two]]
; CHECK-SPIRV: %[[#]] = OpLoad %[[#int]] %[[#c1]]
  %20 = load ptr, ptr %s1, align 8
  %c = getelementptr inbounds %struct.ST, ptr %20, i32 0, i32 2
  %21 = load i32, ptr %c, align 4
; CHECK-SPIRV: %[[#]] = OpBitcast %[[#structPP]]
; CHECK-SPIRV: %[[#t12:]] = OpLoad %[[#structP]]
; CHECK-SPIRV: %[[#c2:]] = OpInBoundsPtrAccessChain %[[#intP]] %[[#t12]] %[[#zero]] %[[#two]]
; CHECK-SPIRV: %[[#]] = OpLoad %[[#int]] %[[#c2]]
  %22 = load ptr, ptr %s2, align 8
  %c11 = getelementptr inbounds %struct.ST, ptr %22, i32 0, i32 2
  %23 = load i32, ptr %c11, align 4
  %sub12 = sub nsw i32 %21, %23
  store i32 %sub12, ptr %retval, align 4
  br label %return

return:                                           ; preds = %if.end10, %if.then6, %if.then
  %24 = load i32, ptr %retval, align 4
  ret i32 %24
}
