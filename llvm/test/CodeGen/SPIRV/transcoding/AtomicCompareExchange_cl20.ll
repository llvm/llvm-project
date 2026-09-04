; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s 
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64v1.2-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-NOT: OpCapability Int64Atomics

; CHECK-DAG: %[[#int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#int8:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#DeviceScope:]] =  OpConstant %[[#int]] 1
; CHECK-DAG: %[[#SequentiallyConsistent_MS:]] = OpConstant %[[#int]] 16
; CHECK-DAG: %[[#int_ptr:]] = OpTypePointer Generic %[[#int]]
; CHECK-DAG: %[[#int_ptr8:]] = OpTypePointer Generic %[[#int8]]
; CHECK-DAG: %[[#bool:]] = OpTypeBool

define spir_func void @test(ptr addrspace(4) %object, ptr addrspace(4) %expected, i32 %desired) {

; CHECK: %[[#object:]] = OpFunctionParameter %[[#int_ptr8]]
; CHECK: %[[#expected:]] = OpFunctionParameter %[[#int_ptr8]] 
; CHECK: %[[#desired:]] = OpFunctionParameter %[[#int]]

entry:
  %object.addr = alloca ptr addrspace(4), align 4
  %expected.addr = alloca ptr addrspace(4), align 4
  %desired.addr = alloca i32, align 4
  %strong_res = alloca i8, align 1
  %res = alloca i8, align 1
  %weak_res = alloca i8, align 1
  store ptr addrspace(4) %object, ptr %object.addr, align 4
  store ptr addrspace(4) %expected, ptr %expected.addr, align 4
  store i32 %desired, ptr %desired.addr, align 4
  %0 = load ptr addrspace(4), ptr %object.addr, align 4
  %1 = load ptr addrspace(4), ptr %expected.addr, align 4
  %2 = load i32, ptr %desired.addr, align 4

; CHECK-DAG: OpStore %[[#object_addr:]] %[[#object]]
; CHECK-DAG: OpStore %[[#expected_addr:]] %[[#expected]]
; CHECK-DAG: OpStore %[[#desired_addr:]] %[[#desired]]
  
; CHECK: %[[#Pointer:]] = OpLoad %[[#int_ptr]] %[[#]]
; CHECK: %[[#exp:]] = OpLoad %[[#int_ptr]] %[[#]]
; CHECK: %[[#Value:]] = OpLoad %[[#int]] %[[#desired_addr]]
; CHECK: %[[#Comparator:]] = OpLoad %[[#int]] %[[#exp]]

; CHECK: %[[#Result:]] = OpAtomicCompareExchange %[[#int]] %[[#]] %[[#DeviceScope]] %[[#SequentiallyConsistent_MS]] %[[#SequentiallyConsistent_MS]] %[[#Value]] %[[#Comparator]]
  %call = call spir_func zeroext i1 @_Z30atomic_compare_exchange_strongPVU3AS4U7_AtomiciPU3AS4ii(ptr addrspace(4) %0, ptr addrspace(4) %1, i32 %2)

; CHECK-NEXT: OpStore %[[#exp]] %[[#Result]]
; CHECK-NEXT: %[[#CallRes:]] = OpIEqual %[[#bool]] %[[#Result]] %[[#Comparator]]
; CHECK-NOT: %[[#Result]]

  %frombool = zext i1 %call to i8
  store i8 %frombool, ptr %strong_res, align 1
  %3 = load i8, ptr %strong_res, align 1
  %tobool = trunc i8 %3 to i1
  %lnot = xor i1 %tobool, true
  %frombool1 = zext i1 %lnot to i8
  store i8 %frombool1, ptr %res, align 1
  %4 = load ptr addrspace(4), ptr %object.addr, align 4
  %5 = load ptr addrspace(4), ptr %expected.addr, align 4
  %6 = load i32, ptr %desired.addr, align 4

; CHECK: %[[#Pointer:]] = OpLoad %[[#int_ptr]] %[[#]]
; CHECK: %[[#exp:]] = OpLoad %[[#int_ptr]] %[[#]]
; CHECK: %[[#Value:]] = OpLoad %[[#int]] %[[#desired_addr]]
; CHECK: %[[#ComparatorWeak:]] = OpLoad %[[#int]] %[[#exp]]

; CHECK: %[[#Result:]] = OpAtomicCompareExchangeWeak %[[#int]] %[[#]] %[[#DeviceScope]] %[[#SequentiallyConsistent_MS]] %[[#SequentiallyConsistent_MS]] %[[#Value]] %[[#ComparatorWeak]]
  %call2 = call spir_func zeroext i1 @_Z28atomic_compare_exchange_weakPVU3AS4U7_AtomiciPU3AS4ii(ptr addrspace(4) %4, ptr addrspace(4) %5, i32 %6)

; CHECK-NEXT: OpStore %[[#exp]] %[[#Result]]
; CHECK-NEXT: %[[#CallRes:]] = OpIEqual %[[#bool]] %[[#Result]] %[[#ComparatorWeak]]
; CHECK-NOT: %[[#Result]]

  %frombool3 = zext i1 %call2 to i8
  store i8 %frombool3, ptr %weak_res, align 1
  %7 = load i8, ptr %weak_res, align 1
  %tobool4 = trunc i8 %7 to i1
  %lnot5 = xor i1 %tobool4, true
  %frombool6 = zext i1 %lnot5 to i8
  store i8 %frombool6, ptr %res, align 1
  ret void
}

declare spir_func zeroext i1 @_Z30atomic_compare_exchange_strongPVU3AS4U7_AtomiciPU3AS4ii(ptr addrspace(4), ptr addrspace(4), i32) #1
declare spir_func zeroext i1 @_Z28atomic_compare_exchange_weakPVU3AS4U7_AtomiciPU3AS4ii(ptr addrspace(4), ptr addrspace(4), i32) #1
