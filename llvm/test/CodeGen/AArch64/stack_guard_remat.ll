; RUN: llc < %s -mtriple=arm64-apple-ios -relocation-model=pic -no-integrated-as | FileCheck %s -check-prefix=DARWIN
; RUN: llc < %s -mtriple=arm64-apple-ios -relocation-model=static -no-integrated-as | FileCheck %s -check-prefix=DARWIN
; RUN: llc < %s -mtriple=aarch64-linux-gnu -relocation-model=pic -no-integrated-as | FileCheck %s -check-prefix=PIC-LINUX
; RUN: llc < %s -mtriple=aarch64-linux-gnu -relocation-model=static -code-model=large -no-integrated-as | FileCheck %s -check-prefix=STATIC-LARGE
; RUN: llc < %s -mtriple=aarch64-linux-gnu -relocation-model=static -code-model=small -no-integrated-as | FileCheck %s -check-prefix=STATIC-SMALL

; RUN: llc < %s -global-isel -global-isel-abort=2 -pass-remarks-missed=gisel* -mtriple=arm64-apple-ios -relocation-model=pic -no-integrated-as 2>&1 | FileCheck %s -check-prefixes=DARWIN,FALLBACK
; RUN: llc < %s -global-isel -global-isel-abort=2 -pass-remarks-missed=gisel* -mtriple=arm64-apple-ios -relocation-model=static -no-integrated-as 2>&1 | FileCheck %s -check-prefixes=DARWIN,FALLBACK
; RUN: llc < %s -global-isel -global-isel-abort=2 -pass-remarks-missed=gisel* -mtriple=aarch64-linux-gnu -relocation-model=pic -no-integrated-as 2>&1 | FileCheck %s -check-prefixes=PIC-LINUX,FALLBACK
; RUN: llc < %s -global-isel -global-isel-abort=2 -pass-remarks-missed=gisel* -mtriple=aarch64-linux-gnu -relocation-model=static -code-model=large -no-integrated-as 2>&1 | FileCheck %s -check-prefixes=STATIC-LARGE,FALLBACK
; RUN: llc < %s -global-isel -global-isel-abort=2 -pass-remarks-missed=gisel* -mtriple=aarch64-linux-gnu -relocation-model=static -code-model=small -no-integrated-as 2>&1 | FileCheck %s -check-prefixes=STATIC-SMALL,FALLBACK

; DARWIN: foo2
; DARWIN: adrp [[R0:x[0-9]+]], ___stack_chk_guard@GOTPAGE
; DARWIN: ldr [[R1:x[0-9]+]], {{\[}}[[R0]], ___stack_chk_guard@GOTPAGEOFF{{\]}}
; DARWIN: ldr {{x[0-9]+}}, {{\[}}[[R1]]{{\]}}

; PIC-LINUX: foo2
; PIC-LINUX: adrp [[R0:x[0-9]+]], :got:__stack_chk_guard
; PIC-LINUX: ldr [[R1:x[0-9]+]], {{\[}}[[R0]], :got_lo12:__stack_chk_guard{{\]}}
; PIC-LINUX: ldr {{x[0-9]+}}, {{\[}}[[R1]]{{\]}}

; STATIC-LARGE: foo2
; STATIC-LARGE: movz [[R0:x[0-9]+]], #:abs_g0_nc:__stack_chk_guard
; STATIC-LARGE: movk [[R0]], #:abs_g1_nc:__stack_chk_guard
; STATIC-LARGE: movk [[R0]], #:abs_g2_nc:__stack_chk_guard
; STATIC-LARGE: movk [[R0]], #:abs_g3:__stack_chk_guard
; STATIC-LARGE: ldr {{x[0-9]+}}, {{\[}}[[R0]]{{\]}}

; STATIC-SMALL: foo2
; STATIC-SMALL: adrp [[R0:x[0-9]+]], __stack_chk_guard
; STATIC-SMALL: ldr {{x[0-9]+}}, {{\[}}[[R0]], :lo12:__stack_chk_guard{{\]}}

; FALLBACK-NOT: remark:{{.*}}llvm.lifetime.end
; FALLBACK-NOT: remark:{{.*}}llvm.lifetime.start
define i32 @test_stack_guard_remat() #0 {
entry:
  %a1 = alloca [256 x i32], align 4
  %0 = bitcast [256 x i32]* %a1 to i8*
  call void @llvm.lifetime.start.p0i8(i64 1024, i8* %0)
  %arraydecay = getelementptr inbounds [256 x i32], [256 x i32]* %a1, i64 0, i64 0
  call void @foo3(i32* %arraydecay)
  call void asm sideeffect "foo2", "~{w0},~{w1},~{w2},~{w3},~{w4},~{w5},~{w6},~{w7},~{w8},~{w9},~{w10},~{w11},~{w12},~{w13},~{w14},~{w15},~{w16},~{w17},~{w18},~{w19},~{w20},~{w21},~{w22},~{w23},~{w24},~{w25},~{w26},~{w27},~{w28},~{w29},~{w30}"()
  call void @llvm.lifetime.end.p0i8(i64 1024, i8* %0)
  ret i32 0
}

; Function Attrs: nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture)

declare void @foo3(i32*)

; Function Attrs: nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture)

attributes #0 = { nounwind sspstrong "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
