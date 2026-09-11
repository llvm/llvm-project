; RUN: llc -mtriple=riscv64-unknown-linux-gnu -stop-after=finalize-isel \
; RUN:   -o - %s | FileCheck %s

declare void @sink_i8(i8 signext)
declare void @sink_i16(i16 signext)
declare void @sink_i32(i32 signext)

define i8 @callbr_i8_signext(i1 %cond) {
; CHECK-LABEL: name: callbr_i8_signext
; CHECK:       INLINEASM_BR
; CHECK-NEXT:  {{.*}}:gpr = COPY
entry:
  %result = callbr i8 asm "", "=&r,!i"()
            to label %normal [label %indirect]

normal:
  ret i8 0

indirect:
  br i1 %cond, label %use, label %exit

use:
  call void @sink_i8(i8 signext %result)
  br label %exit

exit:
  ret i8 0
}

define i16 @callbr_i16_signext(i1 %cond) {
; CHECK-LABEL: name: callbr_i16_signext
; CHECK:       INLINEASM_BR
; CHECK-NEXT:  {{.*}}:gpr = COPY
entry:
  %result = callbr i16 asm "", "=&r,!i"()
            to label %normal [label %indirect]

normal:
  ret i16 0

indirect:
  br i1 %cond, label %use, label %exit

use:
  call void @sink_i16(i16 signext %result)
  br label %exit

exit:
  ret i16 0
}

define i32 @callbr_i32_signext(i1 %cond) {
; CHECK-LABEL: name: callbr_i32_signext
; CHECK:       INLINEASM_BR
; CHECK-NEXT:  {{.*}}:gpr = COPY
entry:
  %result = callbr i32 asm "", "=&r,!i"()
            to label %normal [label %indirect]

normal:
  ret i32 0

indirect:
  br i1 %cond, label %use, label %exit

use:
  call void @sink_i32(i32 signext %result)
  br label %exit

exit:
  ret i32 0
}
