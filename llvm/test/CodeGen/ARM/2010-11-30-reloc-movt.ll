; RUN: llc  %s -mtriple=armv7-linux-gnueabi -filetype=obj -o - | \
; RUN:    elf-dump --dump-section-data | FileCheck  -check-prefix=OBJ %s

target triple = "armv7-none-linux-gnueabi"

@a = external global i8

define arm_aapcs_vfpcc i32 @barf() nounwind {
entry:
  %0 = tail call arm_aapcs_vfpcc  i32 @foo(i8* @a) nounwind
  ret i32 %0
; OBJ:         '.text'
; OBJ-NEXT:    'sh_type'
; OBJ-NEXT:    'sh_flags'
; OBJ-NEXT:    'sh_addr'
; OBJ-NEXT:    'sh_offset'
; OBJ-NEXT:    'sh_size'
; OBJ-NEXT:    'sh_link'
; OBJ-NEXT:    'sh_info'
; OBJ-NEXT:    'sh_addralign'
; OBJ-NEXT:    'sh_entsize'
; OBJ-NEXT:    '_section_data', '00482de9 000000e3 000040e3 feffffeb 0088bde8'

; OBJ:            Relocation 0
; OBJ-NEXT:       'r_offset', 0x00000004
; OBJ-NEXT:       'r_sym', 0x000007
; OBJ-NEXT:        'r_type', 0x2b

; OBJ:          Relocation 1
; OBJ-NEXT:       'r_offset', 0x00000008
; OBJ-NEXT:       'r_sym'
; OBJ-NEXT:        'r_type', 0x2c

; OBJ:          # Relocation 2
; OBJ-NEXT:       'r_offset', 0x0000000c
; OBJ-NEXT:       'r_sym', 0x000008
; OBJ-NEXT:       'r_type', 0x1c

}

declare arm_aapcs_vfpcc i32 @foo(i8*)

