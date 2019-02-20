; RUN: llc -O0 -mtriple=mips-img-linux-gnu -mcpu=mips32r2 < %s -o - | FileCheck %s --check-prefixes=CHECK32
; RUN: llc -O0 -mtriple=mips-img-linux-gnu -mcpu=mips32r6 < %s -o - | FileCheck %s --check-prefixes=CHECK32
; RUN: llc -O0 -mtriple=mips64-img-linux-gnu -mcpu=mips64r2 < %s -o - | FileCheck %s --check-prefixes=CHECK64R2
; RUN: llc -O0 -mtriple=mips64-img-linux-gnu -mcpu=mips64r6 < %s -o - | FileCheck %s --check-prefixes=CHECK64R6

declare i32 @foo(...)

define i32 @boo2(i32 signext %argc) {
; CHECK-LABEL: test_label_2:

; CHECK32: j $BB0_5
; CHECK32-NEXT: nop
; CHECK64R2: j .LBB0_5
; CHECK64R2-NEXT: nop
; CHECK64R6: j .LBB0_5
; CHECK64R6-NEXT: nop

entry:
  %retval = alloca i32, align 4
  %argc.addr = alloca i32, align 4
  store i32 0, i32* %retval, align 4
  store i32 %argc, i32* %argc.addr, align 4
  call void asm sideeffect "test_label_2:", "~{$1}"()
  %0 = load i32, i32* %argc.addr, align 4
  %cmp = icmp sgt i32 %0, 1
  br i1 %cmp, label %if.then, label %if.end

if.then:
  call void asm sideeffect ".space 268435052", "~{$1}"()
  %call = call i32 bitcast (i32 (...)* @foo to i32 ()*)()
  store i32 %call, i32* %retval, align 4
  br label %return

if.end:
  store i32 0, i32* %retval, align 4
  br label %return

return:
  %1 = load i32, i32* %retval, align 4
  ret i32 %1
}
