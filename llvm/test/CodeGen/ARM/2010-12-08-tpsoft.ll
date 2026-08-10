; RUN: llc  %s -mtriple=armv7-linux-gnueabi -o - | \
; RUN:    FileCheck  -check-prefix=ELFASM %s 
; RUN: llc  %s -mtriple=armv7-linux-gnueabi -filetype=obj -o - | \
; RUN:    llvm-readobj -S --sd - | FileCheck  -check-prefix=ELFOBJ %s

;; Make sure that bl __aeabi_read_tp is materialized and fixed up correctly
;; in the obj case. 

@i = external thread_local global i32
@a = external global i8
@b = external global [10 x i8]

define arm_aapcs_vfpcc i32 @main() nounwind {
entry:
  %0 = load i32, ptr @i, align 4
  switch i32 %0, label %bb2 [
    i32 12, label %bb
    i32 13, label %bb1
  ]

bb:                                               ; preds = %entry
  %1 = tail call arm_aapcs_vfpcc  i32 @foo(ptr @a) nounwind
  ret i32 %1
; ELFASM:       	bl	__aeabi_read_tp


; ELFOBJ:      Sections [
; ELFOBJ:        Section {
; ELFOBJ:          Name: .text
; ELFOBJ:          SectionData (
;;;                  BL __aeabi_read_tp is ---------+
;;;                                                 V
; ELFOBJ-NEXT:     0000: 00482DE9 3C009FE5 00109FE7 FEFFFFEB


bb1:                                              ; preds = %entry
  %2 = tail call arm_aapcs_vfpcc  i32 @bar(ptr @b) nounwind
  ret i32 %2

bb2:                                              ; preds = %entry
  ret i32 -1
}

declare arm_aapcs_vfpcc i32 @foo(ptr)

declare arm_aapcs_vfpcc i32 @bar(ptr)
