;Check 5.5 Parameter Passing --> Stage C --> C.1.cp statement for VA functions.
;Note: There are no VFP CPRCs in a variadic procedure.
;Check that after %C was sent to stack, we set Next Core Register Number to R4.

;This test is simplified IR version of
;test-suite/SingleSource/UnitTests/2002-05-02-ManyArguments.c

;RUN: llc -mtriple=thumbv7-linux-gnueabihf -float-abi=hard < %s | FileCheck %s

@.str = private unnamed_addr constant [13 x i8] c"%d %d %f %i\0A\00", align 1

;CHECK-LABEL: printfn:
define void @printfn(i32 %a, i16 signext %b, double %C, i8 signext %E) {
entry:
  %conv = sext i16 %b to i32
  %conv1 = sext i8 %E to i32
  %call = tail call i32 (ptr, ...) @printf(
	ptr @.str, ; --> R0
        i32 %a,                                          ; --> R1
        i32 %conv,                                       ; --> R2
        double %C,                                       ; --> SP, NCRN := R4
;CHECK:    str r2, [sp, #8]                                                                     
        i32 %conv1)                                      ; --> SP+8
  ret void
}

declare i32 @printf(ptr nocapture, ...)

