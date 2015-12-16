; RUN: llc < %s -mtriple=x86_64-unknown-unknown | FileCheck %s

define i32 @shl48sar47(i64 %a) #0 {
; CHECK-LABEL: shl48sar47:
; CHECK:       # BB#0:
; CHECK-NEXT:    movswq %di, %rax
  %1 = shl i64 %a, 48
  %2 = ashr exact i64 %1, 47
  %3 = trunc i64 %2 to i32
  ret i32 %3
}

define i32 @shl48sar49(i64 %a) #0 {
; CHECK-LABEL: shl48sar49:
; CHECK:       # BB#0:
; CHECK-NEXT:    movswq %di, %rax
  %1 = shl i64 %a, 48
  %2 = ashr exact i64 %1, 49
  %3 = trunc i64 %2 to i32
  ret i32 %3
}

define i32 @shl56sar55(i64 %a) #0 {
; CHECK-LABEL: shl56sar55:
; CHECK:       # BB#0:
; CHECK-NEXT:    movsbq %dil, %rax
  %1 = shl i64 %a, 56
  %2 = ashr exact i64 %1, 55
  %3 = trunc i64 %2 to i32
  ret i32 %3
}

define i32 @shl56sar57(i64 %a) #0 {
; CHECK-LABEL: shl56sar57:
; CHECK:       # BB#0:
; CHECK-NEXT:    movsbq %dil, %rax
  %1 = shl i64 %a, 56
  %2 = ashr exact i64 %1, 57
  %3 = trunc i64 %2 to i32
  ret i32 %3
}

attributes #0 = { nounwind }
