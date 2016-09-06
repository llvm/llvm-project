; RUN: llc < %s -mtriple thumbv7-apple-ios -O1 | FileCheck %s

; rdar://15144402
; Make sure we don't assume 4-byte alignment when loading from a byval argument
; with alignment of 2.
; CHECK: ldr r1, [r[[REG:[0-9]+]]]
; CHECK: ldr r2, [r[[REG]], #4]
; CHECK: ldr r3, [r[[REG]], #8]
; CHECK-NOT: ldm
; CHECK: .p2align	1 @ @sID

%struct.ModuleID = type { [32 x i8], [32 x i8], i16 }

@sID = internal constant %struct.ModuleID { [32 x i8] c"TEST\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00", [32 x i8] c"1.0\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00", i16 23 }, align 2

; Function Attrs: nounwind ssp
define void @Client() #0 {
entry:
  tail call void @Logger(i8 signext 97, %struct.ModuleID* byval @sID) #2
  ret void
}

declare void @Logger(i8 signext, %struct.ModuleID* byval) #1

attributes #0 = { nounwind ssp "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }
