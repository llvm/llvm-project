; RUN: llc -mtriple=mipsel -mcpu=mips32r2 -mattr=+micromips \
; RUN:   -relocation-model=static -O2 < %s | FileCheck %s
; RUN: llc -mtriple=mipsel -mcpu=mips32r6 -mattr=+micromips \
; RUN:   -relocation-model=static -O2 < %s | FileCheck %s -check-prefix=CHECK-MMR6

; Function Attrs: nounwind
define i32 @foo(i32 signext %a) #0 {
entry:
  %a.addr = alloca i32, align 4
  store i32 %a, ptr %a.addr, align 4
  %0 = load i32, ptr %a.addr, align 4
  %shl = shl i32 %0, 2
  %call = call i32 @bar(i32 signext %shl)
  ret i32 %call
}

declare i32 @bar(i32 signext) #1

; CHECK:      jals
; CHECK-NEXT: sll16
; CHECK-MMR6: balc
; CHECK-MMR6-NOT: sll16
