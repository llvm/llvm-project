; RUN: opt -S -passes=instcombine %s -o - | FileCheck %s

declare i64 @llrintf(float)

; Positive number test
; CHECK-LABEL: define i64 @test_llrintf_pos()
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i64 4
define i64 @test_llrintf_pos() {
entry:
  %val = call i64 @llrintf(float 3.5)
  ret i64 %val
}

; Negative number test
; CHECK-LABEL: define i64 @test_llrintf_neg()
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i64 -2
define i64 @test_llrintf_neg() {
entry:
  %val = call i64 @llrintf(float -2.5)
  ret i64 %val
}

; Zero test
; CHECK-LABEL: define i64 @test_llrintf_zero()
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i64 0
define i64 @test_llrintf_zero() {
entry:
  %val = call i64 @llrintf(float 0.0)
  ret i64 %val
}
 65 changes: 65 additions & 0 deletions65  
llvm/test/Transforms/InstCombine/llrintl_fp80.ll
Original file line number	Diff line number	Diff line change
@@ -0,0 +1,65 @@
; RUN: opt -S -passes=instcombine %s -o - | FileCheck %s
declare i64 @llrintl(x86_fp80)

; Positive number
; CHECK-LABEL: define i64 @test_llrintl_pos()
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i64 4
define i64 @test_llrintl_pos() {
entry:
  %val = call i64 @llrintl(x86_fp80 0xK4000E000000000000000)
  ret i64 %val
}

; Negative number
; CHECK-LABEL: define i64 @test_llrintl_neg()
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i64 -2
define i64 @test_llrintl_neg() {
entry:
  %val = call i64 @llrintl(x86_fp80 0xKC000A000000000000000)
  ret i64 %val
}

; Zero
; CHECK-LABEL: define i64 @test_llrintl_zero()
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i64 0
define i64 @test_llrintl_zero() {
entry:
  %val = call i64 @llrintl(x86_fp80 0xK00000000000000000000)
  ret i64 %val
}

; NaN
; CHECK-LABEL: define i64 @test_llrintl_nan()
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %val = call i64 @llrintl(x86_fp80 0xK7FFF8000000000000000)
; CHECK-NEXT:    ret i64 %val
define i64 @test_llrintl_nan() {
entry:
  %val = call i64 @llrintl(x86_fp80 0xK7FFF8000000000000000)
  ret i64 %val
}

; +Inf
; CHECK-LABEL: define i64 @test_llrintl_posinf()
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %val = call i64 @llrintl(x86_fp80 0xK7FFF0000000000000000)
; CHECK-NEXT:    ret i64 %val
define i64 @test_llrintl_posinf() {
entry:
  %val = call i64 @llrintl(x86_fp80 0xK7FFF0000000000000000)
  ret i64 %val
}

; -Inf
; CHECK-LABEL: define i64 @test_llrintl_neginf()
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %val = call i64 @llrintl(x86_fp80 0xKFFFF0000000000000000)
; CHECK-NEXT:    ret i64 %val
define i64 @test_llrintl_neginf() {
entry:
  %val = call i64 @llrintl(x86_fp80 0xKFFFF0000000000000000)
  ret i64 %val
}