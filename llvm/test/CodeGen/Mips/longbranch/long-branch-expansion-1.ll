; RUN: llc -O0 -mtriple=mips-img-linux-gnu -mcpu=mips32r6 < %s -o - | FileCheck %s --check-prefixes=CHECK32R6
; RUN: llc -O0 -mtriple=mips64-img-linux-gnu -mcpu=mips64r6 < %s -o - | FileCheck %s --check-prefixes=CHECK64R6

declare i32 @foo(...)

define i32 @boo1(i32 signext %argc) {
; CHECK-LABEL: test_label_1:

; CHECK32R6: j $BB0_3
; CHECK32R6-NEXT: nop
; CHECK64R6: j .LBB0_5
; CHECK64R6-NEXT: nop

entry:
  %retval = alloca i32, align 4
  %argc.addr = alloca i32, align 4
  store i32 0, i32* %retval, align 4
  store i32 %argc, i32* %argc.addr, align 4
  call void asm sideeffect "test_label_1:", "~{$1}"()
  %0 = load i32, i32* %argc.addr, align 4
  %cmp = icmp sgt i32 %0, 1
  br i1 %cmp, label %if.then, label %if.end

if.then:
  call void asm sideeffect ".space 68435052", "~{$1}"()
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
