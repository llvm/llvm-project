; RUN: llc -mtriple=x86_64-linux -o - %s | FileCheck %s

%0 = type { %1, %3* }
%1 = type { %2* }
%2 = type { %2*, i8* }
%3 = type { i32, i32 (i32, i32)* }


declare i32 @Up(...) 
declare i32 @f(i32, i32) 

; check loop block_14 is not merged with block_21
; check loop block_11 is not merged with block_18, block_25
define i32 @foo(%0* nocapture readonly, i32, i1 %c, i8* %p1, %2** %p2) {
; CHECK-LABEL: foo:
; CHECK:     # %block_11
; CHECK-NEXT:        movq    (%r14), %rax
; CHECK-NEXT:        testq   %rax, %rax
; CHECK-NEXT:        je      
; CHECK-NEXT:# %block_14
; CHECK-NEXT:        cmpq    $0, 8(%rax)
; CHECK-NEXT:        jne    
; CHECK-NEXT:# %block_18
; CHECK-NEXT:        movq    (%r14), %rax
; CHECK-NEXT:        testq   %rax, %rax
; CHECK-NEXT:        je    
; CHECK-NEXT:# %block_21
; CHECK-NEXT:# =>This Inner Loop Header
; CHECK-NEXT:        cmpq    $0, 8(%rax)
; CHECK-NEXT:        jne  
; CHECK-NEXT:# %block_25
; CHECK-NEXT:#   in Loop
; CHECK-NEXT:        movq    (%r14), %rax
; CHECK-NEXT:        testq   %rax, %rax
; CHECK-NEXT:        jne 
  br i1 %c, label %block_34, label %block_3

block_3:                                      ; preds = %2
  br i1 %c, label %block_7, label %block_4

block_4:                                      ; preds = %block_3
  %a5 = tail call i32 @f(i32 undef, i32 undef)
  %a6 = icmp eq i32 %a5, 0
  br i1 %a6, label %block_7, label %block_34

block_7:                                      ; preds = %block_4, %block_3
  %a8 = icmp eq %2* null, null
  br i1 %a8, label %block_34, label %block_9

block_9:                                      ; preds = %block_7
  %a10 = icmp eq i8* %p1, null
  br i1 %a10, label %block_11, label %block_32

block_11:                                     ; preds = %block_9
  %a12 = load %2*, %2** %p2, align 8
  %a13 = icmp eq %2* %a12, null
  br i1 %a13, label %block_34, label %block_14

block_14:                                     ; preds = %block_11
  %a15 = getelementptr inbounds %2, %2* %a12, i64 0, i32 1
  %a16 = load i8*, i8** %a15, align 8
  %a17 = icmp eq i8* %a16, null
  br i1 %a17, label %block_18, label %block_32

block_18:                                     ; preds = %block_14
  %a19 = load %2*, %2** %p2, align 8
  %a20 = icmp eq %2* %a19, null
  br i1 %a20, label %block_34, label %block_21

block_21:                                     ; preds = %block_18
  %a22 = getelementptr inbounds %2, %2* %a19, i64 0, i32 1
  %a23 = load i8*, i8** %a22, align 8
  %a24 = icmp eq i8* %a23, null
  br i1 %a24, label %block_25, label %block_32

block_25:                                     ; preds = %block_28, %block_21
  %a26 = load %2*, %2** %p2, align 8
  %a27 = icmp eq %2* %a26, null
  br i1 %a27, label %block_34, label %block_28

block_28:                                     ; preds = %block_25
  %a29 = getelementptr inbounds %2, %2* %a26, i64 0, i32 1
  %a30 = load i8*, i8** %a29, align 8
  %a31 = icmp eq i8* %a30, null
  br i1 %a31, label %block_25, label %block_32

block_32:                                     ; preds = %block_28, %block_21, %block_14, %block_9
  %a33 = tail call i32 (...) @Up()
  br label %block_34

block_34:                                     ; preds = %block_32, %block_25, %block_18, %block_11, %block_7, %block_4, %2
  %a35 = phi i32 [ 0, %2 ], [ %a5, %block_4 ], [ 0, %block_7 ], [ 0, %block_11 ], [ 0, %block_32 ], [ 0, %block_18 ], [ 0, %block_25 ]
  ret i32 %a35
}
