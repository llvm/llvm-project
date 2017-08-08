; RUN: opt -S -partially-inline-libcalls -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck %s
; RUN: opt -S -passes=partially-inline-libcalls -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck %s

define float @f(float %val) {
; CHECK: @f
; CHECK: entry:
; CHECK-NEXT: %[[RES:.+]] = tail call float @sqrtf(float %val) #0
; CHECK-NEXT: %[[CMP:.+]] = fcmp oeq float %[[RES]], %[[RES]]
; CHECK-NEXT: br i1 %[[CMP]], label %[[EXIT:.+]], label %[[CALL:.+]]
; CHECK: [[CALL]]:
; CHECK-NEXT: %[[RES2:.+]] = tail call float @sqrtf(float %val){{$}}
; CHECK-NEXT: br label %[[EXIT]]
; CHECK: [[EXIT]]:
; CHECK-NEXT: %[[RET:.+]] = phi float [ %[[RES]], %entry ], [ %[[RES2]], %[[CALL]] ]
; CHECK-NEXT: ret float %[[RET]]
entry:
  %res = tail call float @sqrtf(float %val)
  ret float %res
}

declare float @sqrtf(float)
