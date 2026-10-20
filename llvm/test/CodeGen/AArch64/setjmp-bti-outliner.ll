; RUN: llc -mtriple=aarch64-none-linux-gnu -enable-machine-outliner < %s | FileCheck %s --check-prefix=BTI
; RUN: llc -mtriple=aarch64-none-linux-gnu -global-isel -enable-machine-outliner < %s | \
; RUN: FileCheck %s --check-prefix=BTI
; RUN: llc -mtriple=aarch64-none-linux-gnu -fast-isel -enable-machine-outliner < %s | \
; RUN: FileCheck %s --check-prefix=BTI
; RUN: llc -mtriple=aarch64-none-linux-gnu -enable-machine-outliner -mattr=+no-bti-at-return-twice < %s | \
; RUN: FileCheck %s --check-prefix=NOBTI
; RUN: llc -mtriple=aarch64-none-linux-gnu -global-isel -enable-machine-outliner -mattr=+no-bti-at-return-twice < %s | \
; RUN: FileCheck %s --check-prefix=NOBTI
; RUN: llc -mtriple=aarch64-none-linux-gnu -fast-isel -enable-machine-outliner -mattr=+no-bti-at-return-twice < %s | \
; RUN: FileCheck %s --check-prefix=NOBTI

; Check that the outliner does not split up the call to setjmp and the bti after it.
; When we do not insert a bti, it is allowed to move the setjmp call into an outlined function.

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
; BTI-LABEL: f:
; BTI:         bl      OUTLINED_FUNCTION_1
; BTI-NEXT:    bl      setjmp
; BTI-NEXT:    hint    #36
; BTI-NEXT:    bl      OUTLINED_FUNCTION_0

; NOBTI:      f:
; NOBTI:        bl      OUTLINED_FUNCTION_0
; NOBTI-NEXT:   bl      OUTLINED_FUNCTION_1

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
; BTI-LABEL: g:
; BTI:         bl      OUTLINED_FUNCTION_1
; BTI-NEXT:    bl      setjmp
; BTI-NEXT:    hint    #36
; BTI-NEXT:    bl      OUTLINED_FUNCTION_0

; NOBTI:      g:
; NOBTI:        bl      OUTLINED_FUNCTION_0
; NOBTI-NEXT:   bl      OUTLINED_FUNCTION_1

entry:
  %call = call i32 @setjmp(ptr noundef null) #1
  %add = add nsw i32 %b, %a
  %mul = mul nsw i32 %add, %a
  %add1 = add nsw i32 %d, %c
  %div = sdiv i32 %mul, %add1
  %add2 = add nsw i32 %div, 2
  ret i32 %add2
}

; NOBTI-LABEL: OUTLINED_FUNCTION_0:
; NOBTI:         b       setjmp
; NOBTI:       OUTLINED_FUNCTION_1:
; NOBTI-LABEL:   ret

attributes #0 = { "branch-target-enforcement" }
attributes #1 = { returns_twice }
