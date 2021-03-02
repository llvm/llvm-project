; RUN: llc -mtriple=armv7a-linux-gnueabihf -verify-machineinstrs %s -o - | FileCheck %s --check-prefix=CHECK-ARM
; RUN: llc -mtriple=armv7a-linux-gnueabihf -verify-machineinstrs -fast-isel %s -o - | FileCheck %s --check-prefix=CHECK-ARM-FAST-ISEL
; RUN: llc -mtriple=armv7a-linux-gnueabihf -verify-machineinstrs -global-isel -global-isel-abort=2 %s -o - | FileCheck %s --check-prefix=CHECK-ARM-GLOBAL-ISEL
; RUN: llc -mtriple=thumbv7a-linux-gnueabihf -verify-machineinstrs %s -o - | FileCheck %s --check-prefix=CHECK-THUMB
; RUN: llc -mtriple=thumbv7a-linux-gnueabihf -verify-machineinstrs -fast-isel %s -o - | FileCheck %s --check-prefix=CHECK-THUMB-FAST-ISEL
; RUN: llc -mtriple=thumbv7a-linux-gnueabihf -verify-machineinstrs -global-isel -global-isel-abort=2 %s -o - | FileCheck %s --check-prefix=CHECK-THUMB-GLOBAL-ISEL

define dso_local void @callee() #0 {
; CHECK-ARM:                    stmdb   sp!, {lr}
; CHECK-ARM-NEXT:               bl      __gnu_mcount_nc
; CHECK-ARM-FAST-ISEL:          stmdb   sp!, {lr}
; CHECK-ARM-FAST-ISEL-NEXT:     bl      __gnu_mcount_nc
; CHECK-ARM-GLOBAL-ISEL:        stmdb   sp!, {lr}
; CHECK-ARM-GLOBAL-ISEL-NEXT:   bl      __gnu_mcount_nc
; CHECK-THUMB:                  push    {lr}
; CHECK-THUMB-NEXT:             bl      __gnu_mcount_nc
; CHECK-THUMB-FAST-ISEL:        push    {lr}
; CHECK-THUMB-FAST-ISEL-NEXT:   bl      __gnu_mcount_nc
; CHECK-THUMB-GLOBAL-ISEL:      push    {lr}
; CHECK-THUMB-GLOBAL-ISEL-NEXT: bl      __gnu_mcount_nc
  call void @llvm.arm.gnu.eabi.mcount()
  ret void
}

define dso_local void @caller() #0 {
; CHECK-ARM:                    stmdb   sp!, {lr}
; CHECK-ARM-NEXT:               bl      __gnu_mcount_nc
; CHECK-ARM-FAST-ISEL:          stmdb   sp!, {lr}
; CHECK-ARM-FAST-ISEL-NEXT:     bl      __gnu_mcount_nc
; CHECK-ARM-GLOBAL-ISEL:        stmdb   sp!, {lr}
; CHECK-ARM-GLOBAL-ISEL-NEXT:   bl      __gnu_mcount_nc
; CHECK-THUMB:                  push    {lr}
; CHECK-THUMB-NEXT:             bl      __gnu_mcount_nc
; CHECK-THUMB-FAST-ISEL:        push    {lr}
; CHECK-THUMB-FAST-ISEL-NEXT:   bl      __gnu_mcount_nc
; CHECK-THUMB-GLOBAL-ISEL:      push    {lr}
; CHECK-THUMB-GLOBAL-ISEL-NEXT: bl      __gnu_mcount_nc
  call void @llvm.arm.gnu.eabi.mcount()
  call void @callee()
  ret void
}

declare void @llvm.arm.gnu.eabi.mcount() #1

attributes #0 = { nofree nounwind }
attributes #1 = { nounwind }
