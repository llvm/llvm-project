; RUN: llc -O0 -opaque-pointers=0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

%struct.ST = type { i32, i32, i32 }

; CHECK-SPIRV: OpName %[[#struct:]] "struct.ST"
; CHECK-SPIRV: %[[#int:]] = OpTypeInt 32 0
; CHECK-SPIRV: %[[#intP:]] = OpTypePointer Function %[[#int]]
; CHECK-SPIRV: %[[#struct]] = OpTypeStruct %[[#int]] %[[#int]] %[[#int]]
; CHECK-SPIRV: %[[#structP:]] = OpTypePointer Function %[[#struct]]
; CHECK-SPIRV: %[[#structPP:]] = OpTypePointer Function %[[#structP]]
; CHECK-SPIRV: %[[#zero:]] = OpConstant %[[#int]] 0
; CHECK-SPIRV: %[[#one:]] = OpConstant %[[#int]] 1
; CHECK-SPIRV: %[[#two:]] = OpConstant %[[#int]] 2

define dso_local spir_func i32 @cmp_func(i8* %p1, i8* %p2) {
entry:
  %retval = alloca i32, align 4
  %p1.addr = alloca i8*, align 8
  %p2.addr = alloca i8*, align 8
; CHECK-SPIRV: %[[#s1:]] = OpVariable %[[#structPP]]
; CHECK-SPIRV: %[[#s2:]] = OpVariable %[[#structPP]]
  %s1 = alloca %struct.ST*, align 8
  %s2 = alloca %struct.ST*, align 8
  store i8* %p1, i8** %p1.addr, align 8
  store i8* %p2, i8** %p2.addr, align 8
  %0 = load i8*, i8** %p1.addr, align 8
; CHECK-SPIRV: %[[#t1:]] = OpBitcast %[[#structP]]
; CHECK-SPIRV: OpStore %[[#s1]] %[[#t1]]
  %1 = bitcast i8* %0 to %struct.ST*
  store %struct.ST* %1, %struct.ST** %s1, align 8
  %2 = load i8*, i8** %p2.addr, align 8
; CHECK-SPIRV: %[[#t2:]] = OpBitcast %[[#structP]]
; CHECK-SPIRV: OpStore %[[#s2]] %[[#t2]]
  %3 = bitcast i8* %2 to %struct.ST*
  store %struct.ST* %3, %struct.ST** %s2, align 8
; CHECK-SPIRV: %[[#t3:]] = OpLoad %[[#structP]] %[[#s1]]
; CHECK-SPIRV: %[[#a1:]] = OpInBoundsPtrAccessChain %[[#intP]] %[[#t3]] %[[#zero]] %[[#zero]]
; CHECK-SPIRV: %[[#]] = OpLoad %[[#int]] %[[#a1]]
  %4 = load %struct.ST*, %struct.ST** %s1, align 8
  %a = getelementptr inbounds %struct.ST, %struct.ST* %4, i32 0, i32 0
  %5 = load i32, i32* %a, align 4
; CHECK-SPIRV: %[[#t4:]] = OpLoad %[[#structP]] %[[#s2]]
; CHECK-SPIRV: %[[#a2:]] = OpInBoundsPtrAccessChain %[[#intP]] %[[#t4]] %[[#zero]] %[[#zero]]
; CHECK-SPIRV: %[[#]] = OpLoad %[[#int]] %[[#a2]]
  %6 = load %struct.ST*, %struct.ST** %s2, align 8
  %a1 = getelementptr inbounds %struct.ST, %struct.ST* %6, i32 0, i32 0
  %7 = load i32, i32* %a1, align 4
  %cmp = icmp ne i32 %5, %7
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
; CHECK-SPIRV: %[[#t5:]] = OpLoad %[[#structP]] %[[#s1]]
; CHECK-SPIRV: %[[#a_1:]] = OpInBoundsPtrAccessChain %[[#intP]] %[[#t5]] %[[#zero]] %[[#zero]]
; CHECK-SPIRV: %[[#]] = OpLoad %[[#int]] %[[#a_1]]
  %8 = load %struct.ST*, %struct.ST** %s1, align 8
  %a2 = getelementptr inbounds %struct.ST, %struct.ST* %8, i32 0, i32 0
  %9 = load i32, i32* %a2, align 4
; CHECK-SPIRV: %[[#t6:]] = OpLoad %[[#structP]] %[[#s2]]
; CHECK-SPIRV: %[[#a_2:]] = OpInBoundsPtrAccessChain %[[#intP]] %[[#t6]] %[[#zero]] %[[#zero]]
; CHECK-SPIRV: %[[#]] = OpLoad %[[#int]] %[[#a_2]]
  %10 = load %struct.ST*, %struct.ST** %s2, align 8
  %a3 = getelementptr inbounds %struct.ST, %struct.ST* %10, i32 0, i32 0
  %11 = load i32, i32* %a3, align 4
  %sub = sub nsw i32 %9, %11
  store i32 %sub, i32* %retval, align 4
  br label %return

if.end:                                           ; preds = %entry
; CHECK-SPIRV: %[[#t7:]] = OpLoad %[[#structP]] %[[#s1]]
; CHECK-SPIRV: %[[#b1:]] = OpInBoundsPtrAccessChain %[[#intP]] %[[#t7]] %[[#zero]] %[[#one]]
; CHECK-SPIRV: %[[#]] = OpLoad %[[#int]] %[[#b1]]
  %12 = load %struct.ST*, %struct.ST** %s1, align 8
  %b = getelementptr inbounds %struct.ST, %struct.ST* %12, i32 0, i32 1
  %13 = load i32, i32* %b, align 4
; CHECK-SPIRV: %[[#t8:]] = OpLoad %[[#structP]] %[[#s2]]
; CHECK-SPIRV: %[[#b2:]] = OpInBoundsPtrAccessChain %[[#intP]] %[[#t8]] %[[#zero]] %[[#one]]
; CHECK-SPIRV: %[[#]] = OpLoad %[[#int]] %[[#b2]]
  %14 = load %struct.ST*, %struct.ST** %s2, align 8
  %b4 = getelementptr inbounds %struct.ST, %struct.ST* %14, i32 0, i32 1
  %15 = load i32, i32* %b4, align 4
  %cmp5 = icmp ne i32 %13, %15
  br i1 %cmp5, label %if.then6, label %if.end10

if.then6:                                         ; preds = %if.end
; CHECK-SPIRV: %[[#t9:]] = OpLoad %[[#structP]] %[[#s1]]
; CHECK-SPIRV: %[[#b_1:]] = OpInBoundsPtrAccessChain %[[#intP]] %[[#t9]] %[[#zero]] %[[#one]]
; CHECK-SPIRV: %[[#]] = OpLoad %[[#int]] %[[#b_1]]
  %16 = load %struct.ST*, %struct.ST** %s1, align 8
  %b7 = getelementptr inbounds %struct.ST, %struct.ST* %16, i32 0, i32 1
  %17 = load i32, i32* %b7, align 4
; CHECK-SPIRV: %[[#t10:]] = OpLoad %[[#structP]] %[[#s2]]
; CHECK-SPIRV: %[[#b_2:]] = OpInBoundsPtrAccessChain %[[#intP]] %[[#t10]] %[[#zero]] %[[#one]]
; CHECK-SPIRV: %[[#]] = OpLoad %[[#int]] %[[#b_2]]
  %18 = load %struct.ST*, %struct.ST** %s2, align 8
  %b8 = getelementptr inbounds %struct.ST, %struct.ST* %18, i32 0, i32 1
  %19 = load i32, i32* %b8, align 4
  %sub9 = sub nsw i32 %17, %19
  store i32 %sub9, i32* %retval, align 4
  br label %return

if.end10:                                         ; preds = %if.end
; CHECK-SPIRV: %[[#t11:]] = OpLoad %[[#structP]] %[[#s1]]
; CHECK-SPIRV: %[[#c1:]] = OpInBoundsPtrAccessChain %[[#intP]] %[[#t11]] %[[#zero]] %[[#two]]
; CHECK-SPIRV: %[[#]] = OpLoad %[[#int]] %[[#c1]]
  %20 = load %struct.ST*, %struct.ST** %s1, align 8
  %c = getelementptr inbounds %struct.ST, %struct.ST* %20, i32 0, i32 2
  %21 = load i32, i32* %c, align 4
; CHECK-SPIRV: %[[#t12:]] = OpLoad %[[#structP]] %[[#s2]]
; CHECK-SPIRV: %[[#c2:]] = OpInBoundsPtrAccessChain %[[#intP]] %[[#t12]] %[[#zero]] %[[#two]]
; CHECK-SPIRV: %[[#]] = OpLoad %[[#int]] %[[#c2]]
  %22 = load %struct.ST*, %struct.ST** %s2, align 8
  %c11 = getelementptr inbounds %struct.ST, %struct.ST* %22, i32 0, i32 2
  %23 = load i32, i32* %c11, align 4
  %sub12 = sub nsw i32 %21, %23
  store i32 %sub12, i32* %retval, align 4
  br label %return

return:                                           ; preds = %if.end10, %if.then6, %if.then
  %24 = load i32, i32* %retval, align 4
  ret i32 %24
}
