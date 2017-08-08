; RUN: llvm-as %s -o /dev/null

; Make sure speculatable is accepted on a call site if the declaration
; is also speculatable.

declare i32 @speculatable() #0

; Make sure this the attribute is accepted on the call site if the
; declaration matches.
define i32 @call_speculatable() {
  %ret = call i32 @speculatable() #0
  ret i32 %ret
}

define float @call_bitcast_speculatable() {
  %ret = call float bitcast (i32()* @speculatable to float()*)() #0
  ret float %ret
}

attributes #0 = { speculatable }
