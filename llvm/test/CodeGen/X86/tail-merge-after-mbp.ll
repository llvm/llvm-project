; RUN: llc -mtriple=x86_64-linux -o - %s | FileCheck %s

%0 = type { %1, %3* }
%1 = type { %2* }
%2 = type { %2*, i8* }
%3 = type { i32, i32 (i32, i32)* }


declare i32 @Up(...) 
declare i32 @f(i32, i32) 

define i32 @foo(%0* nocapture readonly, i32, i1 %c, i8* %p1, %2** %p2) {
; CHECK-LABEL: foo:
; CHECK:     BB#6:
; CHECK-NEXT:        movq    (%r14), %rax
; CHECK-NEXT:        testq   %rax, %rax
; CHECK-NEXT:        je      
; CHECK-NEXT:# BB#7:
; CHECK-NEXT:        cmpq    $0, 8(%rax)
; CHECK-NEXT:        jne    
; CHECK-NEXT:# BB#8:
; CHECK-NEXT:        movq    (%r14), %rax
; CHECK-NEXT:        testq   %rax, %rax
; CHECK-NEXT:        je    
; CHECK-NEXT:LBB0_9:
; CHECK-NEXT:        cmpq    $0, 8(%rax)
; CHECK-NEXT:        jne  
; CHECK-NEXT:# BB#10:
; CHECK-NEXT:        movq    (%r14), %rax
; CHECK-NEXT:        testq   %rax, %rax
; CHECK-NEXT:        jne 
  br i1 %c, label %34, label %3

; <label>:3:                                      ; preds = %2
  br i1 %c, label %7, label %4

; <label>:4:                                      ; preds = %3
  %5 = tail call i32 @f(i32 undef, i32 undef)
  %6 = icmp eq i32 %5, 0
  br i1 %6, label %7, label %34

; <label>:7:                                      ; preds = %4, %3
  %8 = icmp eq %2* null, null
  br i1 %8, label %34, label %9

; <label>:9:                                      ; preds = %7
  %10 = icmp eq i8* %p1, null
  br i1 %10, label %11, label %32

; <label>:11:                                     ; preds = %9
  %12 = load %2*, %2** %p2, align 8
  %13 = icmp eq %2* %12, null
  br i1 %13, label %34, label %14

; <label>:14:                                     ; preds = %11
  %15 = getelementptr inbounds %2, %2* %12, i64 0, i32 1
  %16 = load i8*, i8** %15, align 8
  %17 = icmp eq i8* %16, null
  br i1 %17, label %18, label %32

; <label>:18:                                     ; preds = %14
  %19 = load %2*, %2** %p2, align 8
  %20 = icmp eq %2* %19, null
  br i1 %20, label %34, label %21

; <label>:21:                                     ; preds = %18
  %22 = getelementptr inbounds %2, %2* %19, i64 0, i32 1
  %23 = load i8*, i8** %22, align 8
  %24 = icmp eq i8* %23, null
  br i1 %24, label %25, label %32

; <label>:25:                                     ; preds = %28, %21
  %26 = load %2*, %2** %p2, align 8
  %27 = icmp eq %2* %26, null
  br i1 %27, label %34, label %28

; <label>:28:                                     ; preds = %25
  %29 = getelementptr inbounds %2, %2* %26, i64 0, i32 1
  %30 = load i8*, i8** %29, align 8
  %31 = icmp eq i8* %30, null
  br i1 %31, label %25, label %32

; <label>:32:                                     ; preds = %28, %21, %14, %9
  %33 = tail call i32 (...) @Up()
  br label %34

; <label>:34:                                     ; preds = %32, %25, %18, %11, %7, %4, %2
  %35 = phi i32 [ 0, %2 ], [ %5, %4 ], [ 0, %7 ], [ 0, %11 ], [ 0, %32 ], [ 0, %18 ], [ 0, %25 ]
  ret i32 %35
}
