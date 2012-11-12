; RUN: llc -march=mips64el -filetype=obj -mcpu=mips64r2 < %s -o - | elf-dump --dump-section-data  | FileCheck %s

; Check that the R_MIPS_GOT_DISP relocations were created.

; CHECK:     ('r_type', 0x13)

@shl = global i64 1, align 8
@.str = private unnamed_addr constant [8 x i8] c"0x%llx\0A\00", align 1

define i32 @main() nounwind {
entry:
  %0 = load i64* @shl, align 8
  %call = tail call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([8 x i8]* @.str, i64 0, i64 0), i64 %0) nounwind
  ret i32 0
}

declare i32 @printf(i8* nocapture, ...) nounwind

