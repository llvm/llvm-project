; RUN: llc -mtriple=aarch64-linux-gnu %s -o - | FileCheck %s --check-prefix=LINUX
; RUN: llc -mtriple=aarch64-apple-ios %s -o - | FileCheck %s --check-prefix=IOS
; RUN: llc -mtriple=aarch64-apple-ios %s -o - -global-isel | FileCheck %s --check-prefix=IOS
; RUN: llc -mtriple=aarch64-linux-gnueabihf %s -filetype=obj -o %t
; RUN: llvm-objdump -d %t | FileCheck %s --check-prefix=OBJ

; OBJ-NOT: dmb

define void @fence_singlethread() {
; LINUX-LABEL: fence_singlethread:
; LINUX-NOT: dmb
; LINUX: // COMPILER BARRIER
; LINUX-NOT: dmb

; IOS-LABEL: fence_singlethread:
; IOS-NOT: dmb
; IOS: ; COMPILER BARRIER
; IOS-NOT: dmb

  fence syncscope("singlethread") seq_cst
  ret void
}
