; Test the HLASM streamer on z/OS to ensure there's no GNU syntax anywhere

; RUN: llc < %s -mtriple=s390x-ibm-zos -emit-gnuas-syntax-on-zos=0 | FileCheck %s

@.str = private unnamed_addr constant [10 x i8] c"Hello %s\0A\00", align 2
@Greeting = global ptr @.str, align 8
@.str.1 = private unnamed_addr constant [6 x i8] c"World\00", align 2

; Function Attrs: noinline nounwind optnone
define void @foo() {
; CHECK-LABEL: L#PPA2 DS 0H
; CHECK: DC XL1'03'
; CHECK: DC XL1'E7'
; CHECK: DC XL1'22'
; CHECK: DC XL1'04'
; CHECK: DC A(CELQSTRT-L#PPA2)
; CHECK: DC XL4'00000000'
; CHECK: DC A(L#DVS-L#PPA2)
; CHECK: DC XL4'00000000'
; CHECK: DC XL1'81'
; CHECK: DC XL1'00'
; CHECK: DC XL2'0000'
; CHECK-LABEL: L#PPA1_foo_0 DS 0H
; CHECK: DC XL1'02'
; CHECK: DC XL1'CE'
; CHECK: DC XL2'0300'
; CHECK: DC A(L#PPA2-L#PPA1_foo_0)
; CHECK: DC XL1'80'
; CHECK: DC XL1'80'
; CHECK: DC XL1'00'
; CHECK: DC XL1'81'
; CHECK: DC XL2'0000'
; CHECK: DC A(L#func_end0-L#EPM_foo_0)
; CHECK: DC XL2'0003'
; CHECK: DC XL3'869696'
; CHECK: DC A(L#EPM_foo_0-L#PPA1_foo_0)
; CHECK-LABEL: L#.str DS 0H
; CHECK: DC XL10'48656C6C6F2025730A00'
; CHECK: DS 0B
; CHECK-LABEL: Greeting DS 0H
; CHECK: DC A(L#.str)
; CHECK: DS 0B
; CHECK-LABEL: L#.str.1 DS 0H
; CHECK: DC XL6'576F726C6400'
; CHECK: END
entry:
  %0 = load ptr, ptr @Greeting, align 8
  call void (ptr, ...) @outs(ptr noundef %0, ptr noundef @.str.1)
  ret void
}

declare void @outs(ptr noundef, ...)
