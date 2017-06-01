; RUN: llc < %s -mtriple=x86_64-apple-darwin -asm-verbose=false | FileCheck %s -check-prefix=64BIT
; rdar://7329206

; In 32-bit the partial register stall would degrade performance.

define zeroext i16 @t1(i16 zeroext %c, i16 zeroext %k) nounwind ssp {
entry:
; 32BIT-LABEL:     t1:
; 32BIT:     movw 20(%esp), %ax
; 32BIT-NOT: movw %ax, %cx
; 32BIT:     leal 1(%eax), %ecx

; 64BIT-LABEL:     t1:
; 64BIT-NOT: movw %si, %ax
; 64BIT:     movl %esi, %eax
  %0 = icmp eq i16 %k, %c                         ; <i1> [#uses=1]
  %1 = add i16 %k, 1                              ; <i16> [#uses=3]
  br i1 %0, label %bb, label %bb1

bb:                                               ; preds = %entry
  tail call void @foo(i16 zeroext %1) nounwind
  ret i16 %1

bb1:                                              ; preds = %entry
  ret i16 %1
}

define zeroext i16 @t2(i16 zeroext %c, i16 zeroext %k) nounwind ssp {
entry:
; 32BIT-LABEL:     t2:
; 32BIT:     movw 20(%esp), %ax
; 32BIT-NOT: movw %ax, %cx
; 32BIT:     leal -1(%eax), %ecx

; 64BIT-LABEL:     t2:
; 64BIT-NOT: movw %si, %ax
; 64BIT:     movl %esi, %eax
; 64BIT:     movzwl %ax
  %0 = icmp eq i16 %k, %c                         ; <i1> [#uses=1]
  %1 = add i16 %k, -1                             ; <i16> [#uses=3]
  br i1 %0, label %bb, label %bb1

bb:                                               ; preds = %entry
  tail call void @foo(i16 zeroext %1) nounwind
  ret i16 %1

bb1:                                              ; preds = %entry
  ret i16 %1
}

declare void @foo(i16 zeroext)

define zeroext i16 @t3(i16 zeroext %c, i16 zeroext %k) nounwind ssp {
entry:
; 32BIT-LABEL:     t3:
; 32BIT:     movw 20(%esp), %ax
; 32BIT-NOT: movw %ax, %cx
; 32BIT:     leal 2(%eax), %ecx

; 64BIT-LABEL:     t3:
; 64BIT-NOT: movw %si, %ax
; 64BIT:     movl %esi, %eax
  %0 = add i16 %k, 2                              ; <i16> [#uses=3]
  %1 = icmp eq i16 %k, %c                         ; <i1> [#uses=1]
  br i1 %1, label %bb, label %bb1

bb:                                               ; preds = %entry
  tail call void @foo(i16 zeroext %0) nounwind
  ret i16 %0

bb1:                                              ; preds = %entry
  ret i16 %0
}

define zeroext i16 @t4(i16 zeroext %c, i16 zeroext %k) nounwind ssp {
entry:
; 32BIT-LABEL:     t4:
; 32BIT:     movw 16(%esp), %ax
; 32BIT:     movw 20(%esp), %cx
; 32BIT-NOT: movw %cx, %dx
; 32BIT:     leal (%ecx,%eax), %edx

; 64BIT-LABEL:     t4:
; 64BIT-NOT: movw %si, %ax
; 64BIT:     movl %esi, %eax
  %0 = add i16 %k, %c                             ; <i16> [#uses=3]
  %1 = icmp eq i16 %k, %c                         ; <i1> [#uses=1]
  br i1 %1, label %bb, label %bb1

bb:                                               ; preds = %entry
  tail call void @foo(i16 zeroext %0) nounwind
  ret i16 %0

bb1:                                              ; preds = %entry
  ret i16 %0
}
