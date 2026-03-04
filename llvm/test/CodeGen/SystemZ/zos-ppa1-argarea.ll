; RUN: llc < %s -mtriple=s390x-ibm-zos | FileCheck %s
%struct.LargeStruct_t = type { [33 x i32] }

@GlobLargeS = hidden global %struct.LargeStruct_t zeroinitializer, align 4
@GlobInt = hidden global i32 0, align 4

; === Check that function with small frame does not emit PPA1 Argument Area Length.
define void @fSmallOutArgArea() {
; CHECK-LABEL: L#EPM_fSmallOutArgArea_0 DS 0H
; CHECK: *   Bit 1: 1 = Leaf function
; CHECK: *   Bit 2: 0 = Does not use alloca
; CHECK:  DC XL4'00000008'
; CHECK: fSmallOutArgArea DS 0H
; CHECK: L#PPA1_fSmallOutArgArea_0 DS 0H
; CHECK: * PPA1 Flags 3
; CHECK:  DC XL1'00'
  ret void
}

; === Check that function with large frame does emit PPA1 Argument Area Length.
define void @fLargeOutArgArea() {
; CHECK-LABEL: L#EPM_fLargeOutArgArea_0 DS 0H
; CHECK: *   Bit 1: 0 = Non-leaf function
; CHECK: *   Bit 2: 0 = Does not use alloca
; CHECK:  DC XL4'00000220'
; CHECK: fLargeOutArgArea DS 0H
; CHECK: L#PPA1_fLargeOutArgArea_0 DS 0H
; CHECK: * PPA1 Flags 3
; CHECK: *   Bit 1: 1 = Argument Area Length is in optional area
; CHECK:  DC XL1'40'
; CHECK: * Argument Area Length
; CHECK:  DC XL4'00000140'
  %1 = load [33 x i32], ptr @GlobLargeS, align 4
  call void @fLargeParm([33 x i32] inreg %1)
  ret void
}

; === Check that function with parameter does emit PPA1 Length/4 of parms
define void @fLargeParm([33 x i64] inreg %arr) {
; CHECK-LABEL: L#EPM_fLargeParm_0 DS 0H
; CHECK: * Length/4 of Parms
; CHECK:  DC XL2'0042'
  %1 = extractvalue [33 x i64] %arr, 1
  call void @foo(i64 %1)
  ret void
}

; === Check that function with alloca call does emit PPA1 Argument Area Length.
define hidden void @fHasAlloca() {
; CHECK-LABEL: L#EPM_fHasAlloca_0 DS 0H
; CHECK: *   Bit 2: 1 = Uses alloca
; CHECK: fHasAlloca DS 0H
; CHECK: L#PPA1_fHasAlloca_0 DS 0H
; CHECK: * PPA1 Flags 3
; CHECK: *   Bit 1: 1 = Argument Area Length is in optional area
; CHECK:  DC XL1'40'
; CHECK: * Argument Area Length
; CHECK:  DC XL4'00000040'
  %p = alloca ptr, align 4
  %1 = load i32, ptr @GlobInt, align 4
  %2 = alloca i8, i32 %1, align 8
  store ptr %2, ptr %p, align 4
  ret void
}

declare void @foo(i64)
