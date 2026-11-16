; RUN: llc -mtriple=aarch64-none-linux-gnu -enable-machine-outliner < %s | \
; RUN: FileCheck %s --check-prefixes=CHECK,BTI --implicit-check-not=OUTLINED_FUNCTION
; RUN: llc -mtriple=aarch64-none-linux-gnu -global-isel -enable-machine-outliner < %s | \
; RUN: FileCheck %s --check-prefix=GISEL-BTI
; RUN: llc -mtriple=aarch64-none-linux-gnu -fast-isel -enable-machine-outliner < %s | \
; RUN: FileCheck %s --check-prefixes=CHECK,BTI --implicit-check-not=OUTLINED_FUNCTION
; RUN: llc -mtriple=aarch64-none-linux-gnu -enable-machine-outliner -mattr=+no-bti-at-return-twice < %s | \
; RUN: FileCheck %s --check-prefixes=CHECK,NOBTI --implicit-check-not=OUTLINED_FUNCTION
; RUN: llc -mtriple=aarch64-none-linux-gnu -global-isel -enable-machine-outliner -mattr=+no-bti-at-return-twice < %s | \
; RUN: FileCheck %s --check-prefix=GISEL-NOBTI
; RUN: llc -mtriple=aarch64-none-linux-gnu -fast-isel -enable-machine-outliner -mattr=+no-bti-at-return-twice < %s | \
; RUN: FileCheck %s --check-prefixes=CHECK,NOBTI --implicit-check-not=OUTLINED_FUNCTION

; Functions calling returns_twice are not outlined; setjmp retains its BTI when enabled.
; TODO: GlobalISel does not set exposesReturnsTwice and still allows outlining.

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

; GISEL-BTI-LABEL: f:
; GISEL-BTI:         bl      OUTLINED_FUNCTION_1
; GISEL-BTI-NEXT:    bl      setjmp
; GISEL-BTI-NEXT:    hint    #36
; GISEL-BTI-NEXT:    bl      OUTLINED_FUNCTION_0

; GISEL-NOBTI-LABEL: f:
; GISEL-NOBTI:         bl      OUTLINED_FUNCTION_0
; GISEL-NOBTI-NEXT:    bl      OUTLINED_FUNCTION_1

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

; GISEL-BTI-LABEL: g:
; GISEL-BTI:         bl      OUTLINED_FUNCTION_1
; GISEL-BTI-NEXT:    bl      setjmp
; GISEL-BTI-NEXT:    hint    #36
; GISEL-BTI-NEXT:    bl      OUTLINED_FUNCTION_0

; GISEL-NOBTI-LABEL: g:
; GISEL-NOBTI:         bl      OUTLINED_FUNCTION_0
; GISEL-NOBTI-NEXT:    bl      OUTLINED_FUNCTION_1

entry:
  %call = call i32 @setjmp(ptr noundef null) #1
  %add = add nsw i32 %b, %a
  %mul = mul nsw i32 %add, %a
  %add1 = add nsw i32 %d, %c
  %div = sdiv i32 %mul, %add1
  %add2 = add nsw i32 %div, 2
  ret i32 %add2
}

; GISEL-NOBTI-LABEL: OUTLINED_FUNCTION_0:
; GISEL-NOBTI:         b       setjmp
; GISEL-NOBTI:       OUTLINED_FUNCTION_1:
; GISEL-NOBTI-LABEL:   ret

attributes #0 = { "branch-target-enforcement" }
attributes #1 = { returns_twice }
