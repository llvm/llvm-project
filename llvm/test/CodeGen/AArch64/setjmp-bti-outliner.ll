; RUN: llc -mtriple=aarch64-none-linux-gnu -enable-machine-outliner < %s | \
; RUN: FileCheck %s --check-prefixes=CHECK,BTI --implicit-check-not=OUTLINED_FUNCTION
; RUN: llc -mtriple=aarch64-none-linux-gnu -global-isel -enable-machine-outliner < %s | \
; RUN: FileCheck %s --check-prefixes=CHECK,BTI --implicit-check-not=OUTLINED_FUNCTION
; RUN: llc -mtriple=aarch64-none-linux-gnu -fast-isel -enable-machine-outliner < %s | \
; RUN: FileCheck %s --check-prefixes=CHECK,BTI --implicit-check-not=OUTLINED_FUNCTION
; RUN: llc -mtriple=aarch64-none-linux-gnu -enable-machine-outliner -mattr=+no-bti-at-return-twice < %s | \
; RUN: FileCheck %s --check-prefixes=CHECK,NOBTI --implicit-check-not=OUTLINED_FUNCTION
; RUN: llc -mtriple=aarch64-none-linux-gnu -global-isel -enable-machine-outliner -mattr=+no-bti-at-return-twice < %s | \
; RUN: FileCheck %s --check-prefixes=CHECK,NOBTI --implicit-check-not=OUTLINED_FUNCTION
; RUN: llc -mtriple=aarch64-none-linux-gnu -fast-isel -enable-machine-outliner -mattr=+no-bti-at-return-twice < %s | \
; RUN: FileCheck %s --check-prefixes=CHECK,NOBTI --implicit-check-not=OUTLINED_FUNCTION

; Functions calling returns_twice are not outlined; setjmp retains its BTI when enabled.

; C source
; --------
; extern int setjmp(ptr);
;
; int f(int a, int b, int c, int d) {
;   setjmp(0);
;   return 1 + a * (a + b) / (c + d);
; }
;
; int g(int a, int b, int c, int d) {
;   setjmp(0);
;   return 2 + a * (a + b) / (c + d);
; }

define i32 @f(i32 noundef %a, i32 noundef %b, i32 noundef %c, i32 noundef %d) #0 {
; CHECK-LABEL: f:
; CHECK:       bl      setjmp
; BTI-NEXT:    hint    #36
; NOBTI-NOT:   hint    #36
; CHECK:       ret

entry:
  %call = call i32 @setjmp(ptr noundef null) #1
  %add = add nsw i32 %b, %a
  %mul = mul nsw i32 %add, %a
  %add1 = add nsw i32 %d, %c
  %div = sdiv i32 %mul, %add1
  %add2 = add nsw i32 %div, 1
  ret i32 %add2
}

declare i32 @setjmp(ptr noundef) #0

define i32 @g(i32 noundef %a, i32 noundef %b, i32 noundef %c, i32 noundef %d) #0 {
; CHECK-LABEL: g:
; CHECK:       bl      setjmp
; BTI-NEXT:    hint    #36
; NOBTI-NOT:   hint    #36
; CHECK:       ret

entry:
  %call = call i32 @setjmp(ptr noundef null) #1
  %add = add nsw i32 %b, %a
  %mul = mul nsw i32 %add, %a
  %add1 = add nsw i32 %d, %c
  %div = sdiv i32 %mul, %add1
  %add2 = add nsw i32 %div, 2
  ret i32 %add2
}

attributes #0 = { "branch-target-enforcement" }
attributes #1 = { returns_twice }
