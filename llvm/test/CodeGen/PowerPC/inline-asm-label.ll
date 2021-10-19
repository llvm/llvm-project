; RUN: llc -mcpu=pwr7 -verify-machineinstrs \
; RUN:     -mtriple=powerpc-unknown-aix < %s  | FileCheck %s

; RUN: llc -mcpu=pwr7 -verify-machineinstrs \
; RUN:     -mtriple=powerpc64-unknown-aix < %s | FileCheck %s

; RUN: llc -mcpu=pwr7 -verify-machineinstrs -no-integrated-as \
; RUN:     -mtriple=powerpc64-unknown-aix < %s | FileCheck %s --check-prefix=NOIS


; Function Attrs: noinline nounwind optnone uwtable
define dso_local signext i32 @NoBarrier_CompareAndSwap(i32* %ptr, i32 signext %old_value, i32 signext %new_value) #0 {
; CHECK-LABEL: NoBarrier_CompareAndSwap:
; CHECK:    #APP
; CHECK-NEXT:  L..tmp0:
; CHECK-NEXT:    lwarx 6, 0, 3
; CHECK-NEXT:    cmpw 4, 6
; CHECK-NEXT:    bne- 0, L..tmp1
; CHECK-NEXT:    stwcx. 5, 0, 3
; CHECK-NEXT:    bne- 0, L..tmp0
; CHECK-NEXT:  L..tmp1:

; NOIS-LABEL: NoBarrier_CompareAndSwap:
; NOIS:    #APP
; NOIS-NEXT: 1: lwarx 6, 0, 3
; NOIS-NEXT:    cmpw 4, 6
; NOIS-NEXT:    bne- 2f
; NOIS-NEXT:    stwcx. 5, 0, 3
; NOIS-NEXT:    bne- 1b
; NOIS-NEXT: 2:

entry:
  %ptr.addr = alloca i32*, align 8                                                                                                                                            %old_value.addr = alloca i32, align 4
  %new_value.addr = alloca i32, align 4
  %result = alloca i32, align 4
  store i32* %ptr, i32** %ptr.addr, align 8
  store i32 %old_value, i32* %old_value.addr, align 4
  store i32 %new_value, i32* %new_value.addr, align 4
  %0 = load i32*, i32** %ptr.addr, align 8
  %1 = load i32, i32* %old_value.addr, align 4
  %2 = load i32, i32* %new_value.addr, align 4
  %3 = call i32 asm sideeffect "1:     lwarx $0, $4, $1   \0A\09       cmpw $2, $0             \0A\09       bne- 2f                         \0A\09       stwcx. $3, $4, $1  \0A\09       bne- 1b                         \0A\092:                                     \0A\09", "=&b,b,b,b,i,~{cr0},~{ctr}"(i32* %0, i32 %1, i32 %2, i32 0)
  store i32 %3, i32* %result, align 4
  %4 = load i32, i32* %result, align 4
  ret i32 %4
}

