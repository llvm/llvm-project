; RUN: llc -O0 -mtriple=mips-img-linux-gnu -mcpu=mips32r6 -relocation-model=pic < %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-PIC
; RUN: llc -O0 -mtriple=mips-img-linux-gnu -mcpu=mips32r6 -relocation-model=static < %s -o - | FileCheck %s --check-prefixes=CHECK-STATIC

declare i32 @boo(...)
declare i32 @foo(...)

define i32 @main(i32 signext %argc, i8** %argv) {
; CHECK: main:
; CHECK: # %bb.1:
; CHECK-PIC: addiu
; CHECK-PIC: sw
; CHECK-PIC: lui
; CHECK-PIC: addiu
; CHECK-PIC: balc
; CHECK-PIC: addu
; CHECK-PIC: lw
; CHECK-PIC: addiu
; CHECK-PIC: jrc
; CHECK-PIC: bc
; CHECK-PIC: bnezc
; CHECK-PIC: nop
; CHECK-PIC: bc

; CHECK-STATIC: bc
; CHECK-STATIC: j
; CHECK-STATIC: bnezc
; CHECK-STATIC: nop
; CHECK-STATIC: j
entry:
  %retval = alloca i32, align 4
  %argc.addr = alloca i32, align 4
  %argv.addr = alloca i8**, align 4
  store i32 0, i32* %retval, align 4
  store i32 %argc, i32* %argc.addr, align 4
  store i8** %argv, i8*** %argv.addr, align 4
  %0 = load i32, i32* %argc.addr, align 4
  %cmp = icmp sgt i32 %0, 1
  br i1 %cmp, label %if.then, label %if.end4

if.then:
  call void asm sideeffect ".space 10", "~{$1}"()
  %1 = load i32, i32* %argc.addr, align 4
  %cmp1 = icmp sgt i32 %1, 3
  br i1 %cmp1, label %if.then2, label %if.end

if.then2:
  call void asm sideeffect ".space 10", "~{$1}"()
  %call = call i32 bitcast (i32 (...)* @boo to i32 ()*)()
  store i32 %call, i32* %retval, align 4
  br label %return

if.end:
  call void asm sideeffect ".space 4194228", "~{$1}"()
  %call3 = call i32 bitcast (i32 (...)* @foo to i32 ()*)()
  store i32 %call3, i32* %retval, align 4
  br label %return

if.end4:
  store i32 0, i32* %retval, align 4
  br label %return

return:
  %2 = load i32, i32* %retval, align 4
  ret i32 %2

}
