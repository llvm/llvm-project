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
;; Tests that reloc to _MergedGlobals show up as explicit symbol reloc


target triple = "armv7-none-linux-gnueabi"

@var_tls = thread_local global i32 1
@var_tls_double = thread_local global double 1.000000e+00
@var_static = internal global i32 1
@var_static_double = internal global double 1.000000e+00
@var_global = global i32 1
@var_global_double = global double 1.000000e+00

declare i32 @mystrlen(i8* nocapture %s) nounwind  

declare void @myhextochar(i32 %n, i8* nocapture %buffer)

declare void @__aeabi_read_tp() nounwind 

declare void @__nacl_read_tp() nounwind  

define i32 @main(i32 %argc, i8** nocapture %argv) nounwind {
entry:
  switch i32 %argc, label %bb3 [
    i32 555, label %bb
    i32 6666, label %bb2
  ]

bb:                                               ; preds = %entry
  store volatile i32 11, i32* @var_tls, align 4
  store volatile double 2.200000e+01, double* @var_tls_double, align 8
  store volatile i32 33, i32* @var_static, align 4
  store volatile double 4.400000e+01, double* @var_static_double, align 8
  store volatile i32 55, i32* @var_global, align 4
  store volatile double 6.600000e+01, double* @var_global_double, align 8
  br label %bb3

bb2:                                              ; preds = %entry
  ret i32 add (i32 add (i32 add (i32 ptrtoint (i32* @var_tls to i32), i32 add (i32 ptrtoint (i32* @var_static to i32), i32 ptrtoint (i32* @var_global to i32))), i32 ptrtoint (double* @var_tls_double to i32)), i32 add (i32 ptrtoint (double* @var_static_double to i32), i32 ptrtoint (double* @var_global_double to i32)))

bb3:                                              ; preds = %bb, %entry
  tail call void @exit(i32 55) noreturn nounwind
  unreachable
}

declare void @exit(i32) noreturn nounwind

;; OBJ:          Relocation 1
;; OBJ-NEXT:     'r_offset',
;; OBJ-NEXT:     'r_sym', 0x000002
;; OBJ-NEXT:     'r_type', 0x2b

;; OBJ:         Symbol 2
;; OBJ-NEXT:    '_MergedGlobals'
;; OBJ-NEXT:    'st_value', 0x00000010
