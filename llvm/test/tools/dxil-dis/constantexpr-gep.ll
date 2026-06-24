; RUN: llc --filetype=obj %s -o - | dxil-dis -o - | FileCheck %s
target triple = "dxil-unknown-shadermodel6.7-library"

; CHECK: [[GLOBAL:@.*]] = unnamed_addr addrspace(3) global [10 x i32] zeroinitializer, align 4
@g = local_unnamed_addr addrspace(3) global [10 x i32] zeroinitializer, align 4

define i32 @fn() #0 {
; CHECK-LABEL:  define i32 @fn()
; CHECK-NEXT:   [[LOAD:%.*]] = load i32, i32 addrspace(3)* getelementptr inbounds ([10 x i32], [10 x i32] addrspace(3)* [[GLOBAL]], i32 0, i32 1), align 4
; CHECK-NEXT:   ret i32 [[LOAD]]
;
  %gep = getelementptr [10 x i32], ptr addrspace(3) @g, i32 0, i32 1
  %ld = load i32, ptr addrspace(3) %gep, align 4
  ret i32 %ld
}

define i32 @fn2() #0 {
; CHECK-LABEL:  define i32 @fn2()
; CHECK-NEXT:   [[LOAD:%.*]] = load i32, i32 addrspace(3)* getelementptr inbounds ([10 x i32], [10 x i32] addrspace(3)* [[GLOBAL]], i32 0, i32 2), align 4
; CHECK-NEXT:   ret i32 [[LOAD]]
;
  %ld = load i32, ptr addrspace(3) getelementptr ([10 x i32], ptr addrspace(3) @g, i32 0, i32 2), align 4
  ret i32 %ld
}

define i32 @fn3() #0 {
; CHECK-LABEL:  define i32 @fn3()
; CHECK-NEXT:   [[LOAD:%.*]] = load i32, i32 addrspace(3)* getelementptr inbounds ([10 x i32], [10 x i32] addrspace(3)* [[GLOBAL]], i32 0, i32 3), align 4
; CHECK-NEXT:   ret i32 [[LOAD]]
;
  %ld = load i32, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @g, i32 12), align 4
  ret i32 %ld
}

attributes #0 = { "hlsl.export" }
