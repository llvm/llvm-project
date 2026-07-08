; REQUIRES: asserts

; RUN: opt -S -passes=loop-vectorize -epilogue-tail-folding-policy=dont-fold-tail \
; RUN:   -debug < %s 2>%t | FileCheck %s
; RUN: cat %t | FileCheck --check-prefix=DEFAULT-CM %s

; RUN: opt -S -passes=loop-vectorize -epilogue-tail-folding-policy=prefer-fold-tail \
; RUN:   -debug < %s 2>%t | FileCheck %s
; RUN: cat %t | FileCheck --check-prefix=EPILOGUE-TF-CM %s


target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"
target triple = "aarch64"

; DEFAULT-CM: Cost for VF 2: 6
; DEFAULT-CM: Cost for VF 4: 4
; DEFAULT-CM: Cost for VF 8: 3
; DEFAULT-CM: Cost for VF 16: 3

; EPILOGUE-TF-CM: LV: epilogue tail-folding is enabled
; EPILOGUE-TF-CM: LV: can fold tail by masking.
; EPILOGUE-TF-CM: LV: CM instances: 2
; EPILOGUE-TF-CM: LV: Predicated CM, calculate costs for VF: 2
; EPILOGUE-TF-CM: Cost for VF 2: 6
; EPILOGUE-TF-CM: LV: Predicated CM, calculate costs for VF: 4
; EPILOGUE-TF-CM: Cost for VF 4: 11
; EPILOGUE-TF-CM: LV: Predicated CM, calculate costs for VF: 8
; EPILOGUE-TF-CM: Cost for VF 8: 21
; EPILOGUE-TF-CM: LV: Predicated CM, calculate costs for VF: 16
; EPILOGUE-TF-CM: Cost for VF 16: 41
;
define void @test_epilogue_tf(ptr %A, i64 %n) {
; CHECK-LABEL: @test_epilogue_tf
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i8, ptr %A, i64 %iv
  store i8 1, ptr %arrayidx, align 1
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp ne i64 %iv.next, %n
  br i1 %exitcond, label %for.body, label %exit

exit:
  ret void
}
