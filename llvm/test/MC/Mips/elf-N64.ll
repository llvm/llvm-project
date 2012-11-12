; RUN: llc -filetype=obj -march=mips64el -mcpu=mips64 -disable-mips-delay-filler %s -o - | elf-dump --dump-section-data  | FileCheck %s

; Check for N64 relocation production.
;
; ModuleID = '../hello.c'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:32-i16:16:32-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v64:64:64-n32"
target triple = "mips64el-unknown-linux"

@str = private unnamed_addr constant [12 x i8] c"hello world\00"

define i32 @main() nounwind {
entry:
; Check that the appropriate relocations were created.

; R_MIPS_GPREL16/R_MIPS_SUB/R_MIPS_HI16
; CHECK:     ('r_type3', 0x05)
; CHECK-NEXT:     ('r_type2', 0x18)
; CHECK-NEXT:     ('r_type', 0x07)

; R_MIPS_GPREL16/R_MIPS_SUB/R_MIPS_LO16
; CHECK:     ('r_type3', 0x06)
; CHECK-NEXT:     ('r_type2', 0x18)
; CHECK-NEXT:     ('r_type', 0x07)

; R_MIPS_GOT_OFST/R_MIPS_NONE/R_MIPS_NONE
; CHECK:     ('r_type3', 0x00)
; CHECK-NEXT:     ('r_type2', 0x00)
; CHECK-NEXT:     ('r_type', 0x14)

; R_MIPS_GOT_OFST/R_MIPS_NONE/R_MIPS_NONE
; CHECK:     ('r_type3', 0x00)
; CHECK-NEXT:     ('r_type2', 0x00)
; CHECK-NEXT:     ('r_type', 0x15)

  %puts = tail call i32 @puts(i8* getelementptr inbounds ([12 x i8]* @str, i64 0, i64 0))
  ret i32 0

}
declare i32 @puts(i8* nocapture) nounwind
