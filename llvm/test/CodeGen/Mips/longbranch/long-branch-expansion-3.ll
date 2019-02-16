; RUN: llc -O0 -mtriple=mips-img-linux-gnu -mcpu=mips32r2 -verify-machineinstrs < %s -o - | FileCheck %s --check-prefixes=CHECK32R2
; RUN: llc -O0 -mtriple=mips-img-linux-gnu -mcpu=mips32r6 -verify-machineinstrs < %s -o - | FileCheck %s --check-prefixes=CHECK32R6
; RUN: llc -O0 -mtriple=mips-img-linux-gnu -mcpu=mips32r2 -verify-machineinstrs -mattr=+use-indirect-jump-hazard < %s -o - | FileCheck %s --check-prefixes=CHECK32-IJH
; RUN: llc -O0 -mtriple=mips-img-linux-gnu -mcpu=mips32r6 -verify-machineinstrs -mattr=+use-indirect-jump-hazard < %s -o - | FileCheck %s --check-prefixes=CHECK32-IJH

; RUN: llc -O0 -mtriple=mips64-img-linux-gnu -mcpu=mips64r2 -verify-machineinstrs < %s -o - | FileCheck %s --check-prefixes=CHECK64R2
; RUN: llc -O0 -mtriple=mips64-img-linux-gnu -mcpu=mips64r6 -verify-machineinstrs < %s -o - | FileCheck %s --check-prefixes=CHECK64R6
; RUN: llc -O0 -mtriple=mips64-img-linux-gnu -mcpu=mips64r2 -verify-machineinstrs -mattr=+use-indirect-jump-hazard < %s -o - | FileCheck %s --check-prefixes=CHECK64-IJH
; RUN: llc -O0 -mtriple=mips64-img-linux-gnu -mcpu=mips64r6 -verify-machineinstrs -mattr=+use-indirect-jump-hazard < %s -o - | FileCheck %s --check-prefixes=CHECK64-IJH

declare i32 @foo(...)

define i32 @boo3(i32 signext %argc) {
; CHECK-LABEL: test_label_3:

; CHECK32R2: lui $1, %hi($BB0_4)
; CHECK32R2-NEXT: addiu $1, $1, %lo($BB0_4)
; CHECK32R2-NEXT:  jr $1

; CHECK32R6: lui $1, %hi($BB0_4)
; CHECK32R6-NEXT: addiu $1, $1, %lo($BB0_4)
; CHECK32R6-NEXT:  jrc $1

; CHECK32-IJH: lui $1, %hi($BB0_4)
; CHECK32-IJH-NEXT: addiu $1, $1, %lo($BB0_4)
; CHECK32-IJH-NEXT:  jr.hb  $1

; CHECK64R2: lui $1, %highest(.LBB0_4)
; CHECK64R2-NEXT: daddiu $1, $1, %higher(.LBB0_4)
; CHECK64R2-NEXT: dsll $1, $1, 16
; CHECK64R2-NEXT: daddiu $1, $1, %hi(.LBB0_4)
; CHECK64R2-NEXT: dsll $1, $1, 16
; CHECK64R2-NEXT: daddiu $1, $1, %lo(.LBB0_4)
; CHECK64R2-NEXT: jr $1

; CHECK64R6: lui $1, %highest(.LBB0_4)
; CHECK64R6-NEXT: daddiu $1, $1, %higher(.LBB0_4)
; CHECK64R6-NEXT: dsll $1, $1, 16
; CHECK64R6-NEXT: daddiu $1, $1, %hi(.LBB0_4)
; CHECK64R6-NEXT: dsll $1, $1, 16
; CHECK64R6-NEXT: daddiu $1, $1, %lo(.LBB0_4)
; CHECK64R6-NEXT: jrc $1

; CHECK64-IJH: lui $1, %highest(.LBB0_4)
; CHECK64-IJH-NEXT: daddiu $1, $1, %higher(.LBB0_4)
; CHECK64-IJH-NEXT: dsll $1, $1, 16
; CHECK64-IJH-NEXT: daddiu $1, $1, %hi(.LBB0_4)
; CHECK64-IJH-NEXT: dsll $1, $1, 16
; CHECK64-IJH-NEXT: daddiu $1, $1, %lo(.LBB0_4)
; CHECK64-IJH-NEXT:  jr.hb  $1

entry:
  %retval = alloca i32, align 4
  %argc.addr = alloca i32, align 4
  store i32 0, i32* %retval, align 4
  store i32 %argc, i32* %argc.addr, align 4
  call void asm sideeffect "test_label_3:", "~{$1}"()
  %0 = load i32, i32* %argc.addr, align 4
  %cmp = icmp sgt i32 %0, 1
  br i1 %cmp, label %if.then, label %if.end

if.then:
  call void asm sideeffect ".space 268435452", "~{$1}"()
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
