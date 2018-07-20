; RUN: llc -march=hexagon < %s | FileCheck %s

; CHECK: count_leading_0:
; CHECK: cl0(r0)
define i32 @count_leading_0(i32 %p) #0 {
  %1 = call i32 @llvm.ctlz.i32(i32 %p, i1 false)
  ret i32 %1
}

; CHECK: count_leading_0p:
; CHECK: cl0(r1:0)
define i32 @count_leading_0p(i64 %p) #0 {
  %1 = call i64 @llvm.ctlz.i64(i64 %p, i1 false)
  %2 = trunc i64 %1 to i32
  ret i32 %2
}

; CHECK: count_leading_1:
; CHECK: cl1(r0)
define i32 @count_leading_1(i32 %p) #0 {
  %1 = xor i32 %p, -1
  %2 = call i32 @llvm.ctlz.i32(i32 %1, i1 false)
  ret i32 %2
}

; CHECK: count_leading_1p:
; CHECK: cl1(r1:0)
define i32 @count_leading_1p(i64 %p) #0 {
  %1 = xor i64 %p, -1
  %2 = call i64 @llvm.ctlz.i64(i64 %1, i1 false)
  %3 = trunc i64 %2 to i32
  ret i32 %3
}



; CHECK: count_trailing_0:
; CHECK: ct0(r0)
define i32 @count_trailing_0(i32 %p) #0 {
  %1 = call i32 @llvm.cttz.i32(i32 %p, i1 false)
  ret i32 %1
}

; CHECK: count_trailing_0p:
; CHECK: ct0(r1:0)
define i32 @count_trailing_0p(i64 %p) #0 {
  %1 = call i64 @llvm.cttz.i64(i64 %p, i1 false)
  %2 = trunc i64 %1 to i32
  ret i32 %2
}

; CHECK: count_trailing_1:
; CHECK: ct1(r0)
define i32 @count_trailing_1(i32 %p) #0 {
  %1 = xor i32 %p, -1
  %2 = call i32 @llvm.cttz.i32(i32 %1, i1 false)
  ret i32 %2
}

; CHECK: count_trailing_1p:
; CHECK: ct1(r1:0)
define i32 @count_trailing_1p(i64 %p) #0 {
  %1 = xor i64 %p, -1
  %2 = call i64 @llvm.cttz.i64(i64 %1, i1 false)
  %3 = trunc i64 %2 to i32
  ret i32 %3
}

declare i32 @llvm.ctlz.i32(i32, i1)
declare i64 @llvm.ctlz.i64(i64, i1)
declare i32 @llvm.cttz.i32(i32, i1)
declare i64 @llvm.cttz.i64(i64, i1)

attributes #0 = { nounwind "target-cpu"="hexagonv55" }

