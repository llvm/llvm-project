; DISABLE: llc -filetype=obj -mtriple mips-unknown-linux %s -o - | elf-dump --dump-section-data  | FileCheck %s
; RUN: false
; XFAIL: *

; Check that this is big endian.
; CHECK: ('e_indent[EI_DATA]', 0x02)

; Make sure that a section table (text) entry is correct.
; CHECK:   (('sh_name', 0x{{[0]*}}5) # '.text'
; CHECK-NEXT:   ('sh_type', 0x{{[0]*}}1)
; CHECK-NEXT:   ('sh_flags', 0x{{[0]*}}6)
; CHECK-NEXT:   ('sh_addr', 0x{{[0-9,a-f]+}})
; CHECK-NEXT:   ('sh_offset', 0x{{[0-9,a-f]+}})
; CHECK-NEXT:   ('sh_size', 0x{{[0-9,a-f]+}})
; CHECK-NEXT:   ('sh_link', 0x{{[0]+}})
; CHECK-NEXT:   ('sh_info', 0x{{[0]+}})
; CHECK-NEXT:   ('sh_addralign', 0x{{[0]*}}4)
; CHECK-NEXT:   ('sh_entsize', 0x{{[0]+}})

; See that at least first 3 instructions are correct: GP prologue
; CHECK-NEXT:   ('_section_data', '3c1c0000 279c0000 0399e021 {{[0-9,a-f, ]*}}')

; ModuleID = '../br1.c'
target datalayout = "E-p:32:32:32-i1:8:8-i8:8:32-i16:16:32-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-n32"
target triple = "mips-unknown-linux"

@x = global i32 1, align 4
@str = private unnamed_addr constant [4 x i8] c"goo\00"
@str2 = private unnamed_addr constant [4 x i8] c"foo\00"

define i32 @main() nounwind {
entry:
  %0 = load i32* @x, align 4
  %tobool = icmp eq i32 %0, 0
  br i1 %tobool, label %if.end, label %foo

if.end:                                           ; preds = %entry
  %puts = tail call i32 @puts(i8* getelementptr inbounds ([4 x i8]* @str, i32 0, i32 0))
  br label %foo

foo:                                              ; preds = %entry, %if.end
  %puts2 = tail call i32 @puts(i8* getelementptr inbounds ([4 x i8]* @str2, i32 0, i32 0))
  ret i32 0
}

declare i32 @puts(i8* nocapture) nounwind

