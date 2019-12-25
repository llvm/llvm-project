; RUN: llc -march=mipsel -relocation-model=pic -O0 -fast-isel-abort=3 -mcpu=mips32r2 \
; RUN:     < %s | FileCheck %s
; RUN: llc -march=mipsel -relocation-model=pic -O0 -fast-isel-abort=3 -mcpu=mips32 \
; RUN:     < %s | FileCheck %s

@b = global i32 1, align 4
@i = global i32 0, align 4
@.str = private unnamed_addr constant [5 x i8] c"%i \0A\00", align 1

; Function Attrs: nounwind
define void @br() #0 {
entry:
  %0 = load i32, i32* @b, align 4
  %tobool = icmp eq i32 %0, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  store i32 6754, i32* @i, align 4
  br label %if.end

if.end:                                           ; preds = %entry, %if.then
  ret void
; FIXME: This instruction is redundant.
; CHECK:  xor  $[[REG1:[0-9]+]], ${{[0-9]+}}, $zero
; CHECK:  sltiu  $[[REG2:[0-9]+]], $[[REG1]], 1
; CHECK:  bgtz  $[[REG2]], $BB[[BL:[0-9]+_[0-9]+]]
; CHECK:  nop
; CHECK:  addiu  ${{[0-9]+}}, $zero, 6754
; CHECK:  sw  ${{[0-9]+}}, 0(${{[0-9]+}})
; CHECK: $BB[[BL]]:

}

attributes #0 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
