; RUN: llc -mtriple=hexagon -relocation-model=pic < %s | FileCheck %s

; Force the use of R14 (by clobbering everything else in the inline asm).
; Make sure that R14 is not set before the __save call (which will clobber
; R14, R15 and R28).
; CHECK: call __save_r16_through_r27
; CHECK: }
; CHECK: r14 =

@.str = private unnamed_addr constant [4 x i8] c"%d\0A\00", align 1

; Function Attrs: nounwind optsize
define i32 @_Z7testR14Pi(ptr nocapture %res) #0 {
entry:
  %0 = load i32, ptr %res, align 4
  %1 = tail call { i32, i32 } asm "r0=$2\0A\09$1=add(r0,#$3)\0A\09$0=add(r0,#$4)\0A\09", "=r,=r,r,i,i,~{r0},~{r1},~{r2},~{r3},~{r4},~{r5},~{r6},~{r7},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r15},~{r16},~{r17},~{r18},~{r19},~{r20},~{r21},~{r22},~{r23},~{r24},~{r25},~{r26},~{r27}"(i32 %0, i32 40, i32 50) #1
  %asmresult = extractvalue { i32, i32 } %1, 0
  %asmresult1 = extractvalue { i32, i32 } %1, 1
  store i32 %asmresult, ptr %res, align 4
  %call = tail call i32 (ptr, ...) @printf(ptr @.str, i32 %asmresult1) #2
  %2 = load i32, ptr %res, align 4
  %call2 = tail call i32 (ptr, ...) @printf(ptr @.str, i32 %2) #2
  ret i32 0
}

; Function Attrs: nounwind optsize
declare i32 @printf(ptr, ...) #0

; Same as above for R15.
; CHECK: call __save_r16_through_r27
; CHECK: }
; CHECK: r15 =

; Function Attrs: nounwind optsize
define i32 @_Z7testR15Pi(ptr nocapture %res) #0 {
entry:
  %0 = load i32, ptr %res, align 4
  %1 = tail call { i32, i32 } asm "r0=$2\0A\09$1=add(r0,#$3)\0A\09$0=add(r0,#$4)\0A\09", "=r,=r,r,i,i,~{r0},~{r1},~{r2},~{r3},~{r4},~{r5},~{r6},~{r7},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r16},~{r17},~{r18},~{r19},~{r20},~{r21},~{r22},~{r23},~{r24},~{r25},~{r26},~{r27}"(i32 %0, i32 40, i32 50) #1
  %asmresult = extractvalue { i32, i32 } %1, 0
  %asmresult1 = extractvalue { i32, i32 } %1, 1
  store i32 %asmresult, ptr %res, align 4
  %call = tail call i32 (ptr, ...) @printf(ptr @.str, i32 %asmresult1) #2
  %2 = load i32, ptr %res, align 4
  %call2 = tail call i32 (ptr, ...) @printf(ptr @.str, i32 %2) #2
  ret i32 0
}

; Same as above for R28.
; CHECK: call __save_r16_through_r27
; CHECK: }
; CHECK: r28 =

; Function Attrs: nounwind optsize
define i32 @_Z7testR28Pi(ptr nocapture %res) #0 {
entry:
  %0 = load i32, ptr %res, align 4
  %1 = tail call { i32, i32 } asm "r0=$2\0A\09$1=add(r0,#$3)\0A\09$0=add(r0,#$4)\0A\09", "=r,=r,r,i,i,~{r0},~{r1},~{r2},~{r3},~{r4},~{r5},~{r6},~{r7},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15},~{r16},~{r17},~{r18},~{r19},~{r20},~{r21},~{r22},~{r23},~{r24},~{r25},~{r26}"(i32 %0, i32 40, i32 50) #1
  %asmresult = extractvalue { i32, i32 } %1, 0
  %asmresult1 = extractvalue { i32, i32 } %1, 1
  store i32 %asmresult, ptr %res, align 4
  %call = tail call i32 (ptr, ...) @printf(ptr @.str, i32 %asmresult1) #2
  %2 = load i32, ptr %res, align 4
  %call2 = tail call i32 (ptr, ...) @printf(ptr @.str, i32 %2) #2
  ret i32 0
}

attributes #0 = { nounwind optsize "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }
attributes #2 = { nounwind optsize }
