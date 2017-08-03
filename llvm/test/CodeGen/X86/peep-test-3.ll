; RUN: llc < %s -mcpu=generic -march=x86 -post-RA-scheduler=false | FileCheck %s
; rdar://7226797

; LLVM should omit the testl and use the flags result from the orl.

; CHECK-LABEL: or:
define void @or(float* %A, i32 %IA, i32 %N) nounwind {
entry:
  %0 = ptrtoint float* %A to i32                  ; <i32> [#uses=1]
  %1 = and i32 %0, 3                              ; <i32> [#uses=1]
  %2 = xor i32 %IA, 1                             ; <i32> [#uses=1]
; CHECK:      orl %e
; CHECK-NEXT: je
  %3 = or i32 %2, %1                              ; <i32> [#uses=1]
  %4 = icmp eq i32 %3, 0                          ; <i1> [#uses=1]
  br i1 %4, label %return, label %bb

bb:                                               ; preds = %entry
  store float 0.000000e+00, float* %A, align 4
  ret void

return:                                           ; preds = %entry
  ret void
}
; CHECK-LABEL: xor:
define void @xor(float* %A, i32 %IA, i32 %N) nounwind {
entry:
  %0 = ptrtoint float* %A to i32                  ; <i32> [#uses=1]
  %1 = and i32 %0, 3                              ; <i32> [#uses=1]
; CHECK:      xorl $1, %e
; CHECK-NEXT: je
  %2 = xor i32 %IA, 1                             ; <i32> [#uses=1]
  %3 = xor i32 %2, %1                              ; <i32> [#uses=1]
  %4 = icmp eq i32 %3, 0                          ; <i1> [#uses=1]
  br i1 %4, label %return, label %bb

bb:                                               ; preds = %entry
  store float 0.000000e+00, float* %A, align 4
  ret void

return:                                           ; preds = %entry
  ret void
}
; CHECK-LABEL: and:
define void @and(float* %A, i32 %IA, i32 %N, i8* %p) nounwind {
entry:
  store i8 0, i8* %p
  %0 = ptrtoint float* %A to i32                  ; <i32> [#uses=1]
  %1 = and i32 %0, 3                              ; <i32> [#uses=1]
  %2 = xor i32 %IA, 1                             ; <i32> [#uses=1]
; CHECK:      andl  $3, %
; CHECK-NEXT: movb  %
; CHECK-NEXT: je
  %3 = and i32 %2, %1                              ; <i32> [#uses=1]
  %t = trunc i32 %3 to i8
  store i8 %t, i8* %p
  %4 = icmp eq i32 %3, 0                          ; <i1> [#uses=1]
  br i1 %4, label %return, label %bb

bb:                                               ; preds = %entry
  store float 0.000000e+00, float* null, align 4
  ret void

return:                                           ; preds = %entry
  ret void
}

; Just like @and, but without the trunc+store. This should use a testb
; instead of an andl.
; CHECK-LABEL: test:
define void @test(float* %A, i32 %IA, i32 %N, i8* %p) nounwind {
entry:
  store i8 0, i8* %p
  %0 = ptrtoint float* %A to i32                  ; <i32> [#uses=1]
  %1 = and i32 %0, 3                              ; <i32> [#uses=1]
  %2 = xor i32 %IA, 1                             ; <i32> [#uses=1]
; CHECK:      testb $3, %
; CHECK-NEXT: je
  %3 = and i32 %2, %1                              ; <i32> [#uses=1]
  %4 = icmp eq i32 %3, 0                          ; <i1> [#uses=1]
  br i1 %4, label %return, label %bb

bb:                                               ; preds = %entry
  store float 0.000000e+00, float* null, align 4
  ret void

return:                                           ; preds = %entry
  ret void
}
