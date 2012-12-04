; RUN: llc -filetype=obj -mtriple mipsel-unknown-linux -mxgot %s -o - | elf-dump --dump-section-data  | FileCheck %s

@.str = private unnamed_addr constant [16 x i8] c"ext_1=%d, i=%d\0A\00", align 1
@ext_1 = external global i32

define void @fill() nounwind {
entry:

; Check that the appropriate relocations were created. 
; For the xgot case we want to see R_MIPS_[GOT|CALL]_[HI|LO]16.

; R_MIPS_HI16
; CHECK:     ('r_type', 0x05)

; R_MIPS_LO16
; CHECK:     ('r_type', 0x06)

; R_MIPS_GOT_HI16
; CHECK:     ('r_type', 0x16)

; R_MIPS_GOT_LO16
; CHECK:     ('r_type', 0x17)

; R_MIPS_GOT
; CHECK:     ('r_type', 0x09)

; R_MIPS_LO16
; CHECK:     ('r_type', 0x06)

; R_MIPS_CALL_HI16
; CHECK:     ('r_type', 0x1e)

; R_MIPS_CALL_LO16
; CHECK:     ('r_type', 0x1f)

  %0 = load i32* @ext_1, align 4
  %call = tail call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([16 x i8]* @.str, i32 0, i32 0), i32 %0) nounwind
  ret void
}

declare i32 @printf(i8* nocapture, ...) nounwind

