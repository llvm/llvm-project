; RUN: opt < %s -S -early-cse | FileCheck %s
; RUN: opt < %s -S -basicaa -early-cse-memssa | FileCheck %s

; CHECK-LABEL: @test1(
define void @test1(float %A, float %B, float* %PA, float* %PB) {
  ; CHECK-NEXT: fadd
  ; CHECK-NEXT: store
  ; CHECK-NEXT: store
  ; CHECK-NEXT: ret
  %C = fadd float %A, %B
  store float %C, float* %PA
  %D = fadd float %B, %A
  store float %D, float* %PB
  ret void
}

; CHECK-LABEL: @test2(
define void @test2(float %A, float %B, i1* %PA, i1* %PB) {
  ; CHECK-NEXT: fcmp
  ; CHECK-NEXT: store
  ; CHECK-NEXT: store
  ; CHECK-NEXT: ret
  %C = fcmp oeq float %A, %B
  store i1 %C, i1* %PA
  %D = fcmp oeq float %B, %A
  store i1 %D, i1* %PB
  ret void
}

; CHECK-LABEL: @test3(
define void @test3(float %A, float %B, i1* %PA, i1* %PB) {
  ; CHECK-NEXT: fcmp
  ; CHECK-NEXT: store
  ; CHECK-NEXT: store
  ; CHECK-NEXT: ret
  %C = fcmp uge float %A, %B
  store i1 %C, i1* %PA
  %D = fcmp ule float %B, %A
  store i1 %D, i1* %PB
  ret void
}

; CHECK-LABEL: @test4(
define void @test4(i32 %A, i32 %B, i1* %PA, i1* %PB) {
  ; CHECK-NEXT: icmp
  ; CHECK-NEXT: store
  ; CHECK-NEXT: store
  ; CHECK-NEXT: ret
  %C = icmp eq i32 %A, %B
  store i1 %C, i1* %PA
  %D = icmp eq i32 %B, %A
  store i1 %D, i1* %PB
  ret void
}

; CHECK-LABEL: @test5(
define void @test5(i32 %A, i32 %B, i1* %PA, i1* %PB) {
  ; CHECK-NEXT: icmp
  ; CHECK-NEXT: store
  ; CHECK-NEXT: store
  ; CHECK-NEXT: ret
  %C = icmp sgt i32 %A, %B
  store i1 %C, i1* %PA
  %D = icmp slt i32 %B, %A
  store i1 %D, i1* %PB
  ret void
}
