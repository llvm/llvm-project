; RUN: opt -S -passes='dxil-legalize' -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s

define float @load() {
; CHECK-LABEL: define float @load
; CHECK-NEXT:    [[ALLOCA:%.*]] = alloca [2 x float], align 4
; CHECK-NEXT:    [[GEP:%.*]] = getelementptr inbounds nuw [2 x float], ptr [[ALLOCA]], i32 0, i32 0
; CHECK-NEXT:    [[LOAD:%.*]] = load float, ptr [[GEP]], align 4
; CHECK-NEXT:    ret float [[LOAD]]
  %a = alloca [2 x float], align 4
  %b = load float, ptr %a, align 4
  ret float %b
}

define void @store() {
; CHECK-LABEL: define void @store
; CHECK-NEXT:    [[ALLOCA:%.*]] = alloca [3 x i32], align 4
; CHECK-NEXT:    [[GEP:%.*]] = getelementptr inbounds nuw [3 x i32], ptr [[ALLOCA]], i32 0, i32 0
; CHECK-NEXT:    store i32 0, ptr [[GEP]], align 4
; CHECK-NEXT:    ret void
  %a = alloca [3 x i32], align 4
  store i32 0, ptr %a, align 4
  ret void
}
