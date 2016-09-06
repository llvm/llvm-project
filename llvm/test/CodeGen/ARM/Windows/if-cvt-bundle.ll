; RUN: llc -mtriple thumbv7--windows-itanium -filetype asm -o - %s | FileCheck %s

declare void @llvm.trap()
declare arm_aapcs_vfpcc zeroext i1 @g()

define arm_aapcs_vfpcc i8* @f() {
entry:
  %call = tail call arm_aapcs_vfpcc zeroext i1 @g()
  br i1 %call, label %if.then, label %if.end

if.then:
  ret i8* bitcast (i1 ()* @g to i8*)

if.end:
  tail call void @llvm.trap()
  unreachable
}

; CHECK: push.w {r11, lr}
; CHECK: bl g
; CHECK: movw [[REG:r[0-9]+]], :lower16:g
; CHECK: movt [[REG]], :upper16:g
; CHECK: pop.w {r11, pc}

