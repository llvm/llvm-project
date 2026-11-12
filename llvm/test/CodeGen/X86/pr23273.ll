; RUN: llc -mtriple=i386-unknown-unknown -mcpu=generic -mattr=-sse2 -fast-isel < %s

; Verify that the backend doesn't crash during fast-isel with an assertion
; failure when selecting a int-to-double conversion. The fast selection routine
; for SINT_TO_FP wrongly assumed that the target had at least SSE2.

@a = common global i32 0, align 4

define i32 @pr23273() {
entry:
  %0 = load i32, ptr @a, align 4
  %conv = sitofp i32 %0 to double
  %call = call i32 @fn1(double %conv)
  ret i32 0
}

declare i32 @fn1(double) #1
