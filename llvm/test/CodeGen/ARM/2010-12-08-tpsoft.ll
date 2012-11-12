; RUN: llc  %s -mtriple=armv7-linux-gnueabi -o - | \
; RUN:    FileCheck  -check-prefix=ELFASM %s 
; RUN: llc  %s -mtriple=armv7-linux-gnueabi -filetype=obj -o - | \
; RUN:    elf-dump --dump-section-data | FileCheck  -check-prefix=ELFOBJ %s

;; Make sure that bl __aeabi_read_tp is materiazlied and fixed up correctly
;; in the obj case. 

@i = external thread_local global i32
@a = external global i8
@b = external global [10 x i8]

define arm_aapcs_vfpcc i32 @main() nounwind {
entry:
  %0 = load i32* @i, align 4
  switch i32 %0, label %bb2 [
    i32 12, label %bb
    i32 13, label %bb1
  ]

bb:                                               ; preds = %entry
  %1 = tail call arm_aapcs_vfpcc  i32 @foo(i8* @a) nounwind
  ret i32 %1
; ELFASM:       	bl	__aeabi_read_tp


; ELFOBJ:   '.text'
; ELFOBJ-NEXT:  'sh_type'
; ELFOBJ-NEXT:  'sh_flags'
; ELFOBJ-NEXT:  'sh_addr'
; ELFOBJ-NEXT:  'sh_offset'
; ELFOBJ-NEXT:  'sh_size'
; ELFOBJ-NEXT:  'sh_link'
; ELFOBJ-NEXT:  'sh_info'
; ELFOBJ-NEXT:  'sh_addralign'
; ELFOBJ-NEXT:  'sh_entsize'
;;;               BL __aeabi_read_tp is ---+
;;;                                        V
; ELFOBJ-NEXT:  00482de9 3c009fe5 00109fe7 feffffeb


bb1:                                              ; preds = %entry
  %2 = tail call arm_aapcs_vfpcc  i32 @bar(i32* bitcast ([10 x i8]* @b to i32*)) nounwind
  ret i32 %2

bb2:                                              ; preds = %entry
  ret i32 -1
}

declare arm_aapcs_vfpcc i32 @foo(i8*)

declare arm_aapcs_vfpcc i32 @bar(i32*)
