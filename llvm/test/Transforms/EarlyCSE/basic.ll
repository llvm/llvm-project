; RUN: opt < %s -S -early-cse | FileCheck %s


; CHECK: @test1
define void @test1(i8 %V, i32 *%P) {
  %A = bitcast i64 42 to double  ;; dead
  %B = add i32 4, 19             ;; constant folds
  store i32 %B, i32* %P
  ; CHECK-NEXT: store i32 23, i32* %P
  
  %C = zext i8 %V to i32
  %D = zext i8 %V to i32  ;; CSE
  store volatile i32 %C, i32* %P
  store volatile i32 %D, i32* %P
  ; CHECK-NEXT: %C = zext i8 %V to i32
  ; CHECK-NEXT: store volatile i32 %C
  ; CHECK-NEXT: store volatile i32 %C
  
  %E = add i32 %C, %C
  %F = add i32 %C, %C
  store volatile i32 %E, i32* %P
  store volatile i32 %F, i32* %P
  ; CHECK-NEXT: %E = add i32 %C, %C
  ; CHECK-NEXT: store volatile i32 %E
  ; CHECK-NEXT: store volatile i32 %E

  %G = add nuw i32 %C, %C         ;; not a CSE with E
  store volatile i32 %G, i32* %P
  ; CHECK-NEXT: %G = add nuw i32 %C, %C
  ; CHECK-NEXT: store volatile i32 %G
  ret void
}


;; Simple load value numbering.
; CHECK: @test2
define i32 @test2(i32 *%P) {
  %V1 = load i32* %P
  %V2 = load i32* %P
  %Diff = sub i32 %V1, %V2
  ret i32 %Diff
  ; CHECK: ret i32 0
}

;; Cross block load value numbering.
; CHECK: @test3
define i32 @test3(i32 *%P, i1 %Cond) {
  %V1 = load i32* %P
  br i1 %Cond, label %T, label %F
T:
  store i32 4, i32* %P
  ret i32 42
F:
  %V2 = load i32* %P
  %Diff = sub i32 %V1, %V2
  ret i32 %Diff
  ; CHECK: F:
  ; CHECK: ret i32 0
}

;; Cross block load value numbering stops when stores happen.
; CHECK: @test4
define i32 @test4(i32 *%P, i1 %Cond) {
  %V1 = load i32* %P
  br i1 %Cond, label %T, label %F
T:
  ret i32 42
F:
  ; Clobbers V1
  store i32 42, i32* %P
  
  %V2 = load i32* %P
  %Diff = sub i32 %V1, %V2
  ret i32 %Diff
  ; CHECK: F:
  ; CHECK: ret i32 %Diff
}

declare i32 @func(i32 *%P) readonly

;; Simple call CSE'ing.
; CHECK: @test5
define i32 @test5(i32 *%P) {
  %V1 = call i32 @func(i32* %P)
  %V2 = call i32 @func(i32* %P)
  %Diff = sub i32 %V1, %V2
  ret i32 %Diff
  ; CHECK: ret i32 0
}

;; Trivial Store->load forwarding
; CHECK: @test6
define i32 @test6(i32 *%P) {
  store i32 42, i32* %P
  %V1 = load i32* %P
  ret i32 %V1
  ; CHECK: ret i32 42
}

;; Trivial dead store elimination.
; CHECK: @test7
define void @test7(i32 *%P) {
  store i32 42, i32* %P
  store i32 45, i32* %P
  ret void
  ; CHECK-NEXT: store i32 45
  ; CHECK-NEXT: ret void
}

;; Readnone functions aren't invalidated by stores.
; CHECK: @test8
define i32 @test8(i32 *%P) {
  %V1 = call i32 @func(i32* %P) readnone
  store i32 4, i32* %P
  %V2 = call i32 @func(i32* %P) readnone
  %Diff = sub i32 %V1, %V2
  ret i32 %Diff
  ; CHECK: ret i32 0
}


