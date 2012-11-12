;; RUN: llc -mtriple=armv7-linux-gnueabi -O3  \
;; RUN:    -mcpu=cortex-a8 -mattr=-neon -mattr=+vfp2  -arm-reserve-r9  \
;; RUN:    -filetype=obj %s -o - | \
;; RUN:   elf-dump --dump-section-data | FileCheck -check-prefix=OBJ %s

;; FIXME: This file needs to be in .s form!
;; The args to llc are there to constrain the codegen only.
;; 
;; Ensure no regression on ARM/gcc compatibility for 
;; emitting explicit symbol relocs for nonexternal symbols 
;; versus section symbol relocs (with offset) - 
;;
;; Default llvm behavior is to emit as section symbol relocs nearly
;; everything that is not an undefined external. Unfortunately, this 
;; diverges from what codesourcery ARM/gcc does!
;;
;; Tests that reloc to .L.str* show up as explicit symbols

target triple = "armv7-none-linux-gnueabi"

@.str = private constant [7 x i8] c"@null\0A\00", align 4
@.str1 = private constant [8 x i8] c"@write\0A\00", align 4
@.str2 = private constant [13 x i8] c"hello worldn\00", align 4
@.str3 = private constant [7 x i8] c"@exit\0A\00", align 4

declare i32 @mystrlen(i8* nocapture %s) nounwind readonly 

declare void @myhextochar(i32 %n, i8* nocapture %buffer) nounwind 

define i32 @main() nounwind {
entry:
  %0 = tail call i32 (...)* @write(i32 1, i8* getelementptr inbounds ([7 x i8]* @.str, i32 0, i32 0), i32 6) nounwind
  %1 = tail call i32 (...)* @write(i32 1, i8* getelementptr inbounds ([8 x i8]* @.str1, i32 0, i32 0), i32 7) nounwind
  %2 = tail call i32 (...)* @write(i32 1, i8* getelementptr inbounds ([13 x i8]* @.str2, i32 0, i32 0), i32 12) nounwind
  %3 = tail call i32 (...)* @write(i32 1, i8* getelementptr inbounds ([7 x i8]* @.str3, i32 0, i32 0), i32 6) nounwind
  tail call void @exit(i32 55) noreturn nounwind
  unreachable
}

declare i32 @write(...)

declare void @exit(i32) noreturn nounwind

;; OBJ:        Relocation 0
;; OBJ-NEXT:    'r_offset', 
;; OBJ-NEXT:    'r_sym', 0x000002
;; OBJ-NEXT:    'r_type', 0x2b

;; OBJ:          Symbol 2
;; OBJ-NEXT:    '.L.str'
