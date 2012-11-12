; RUN: llc -filetype=obj -mtriple x86_64-pc-linux-gnu %s -o - | elf-dump | FileCheck -check-prefix=64 %s

; Test that constant mergeable strings have sh_entsize set.

@.str1 = private unnamed_addr constant [6 x i8] c"tring\00"
@.str2 = private unnamed_addr constant [7 x i8] c"String\00"
@.c8a = private unnamed_addr constant [1 x i64] [i64 42]
@.c8b = private unnamed_addr constant [1 x i64] [i64 42]

define i32 @main() nounwind {
  %1 = call i32 @puts(i8* getelementptr inbounds ([6 x i8]* @.str1, i32 0, i32 0))
  %2 = call i32 @puts(i8* getelementptr inbounds ([7 x i8]* @.str2, i32 0, i32 0))
  call void @foo(i64* getelementptr inbounds ([1 x i64]* @.c8a, i32 0, i32 0))
  call void @foo(i64* getelementptr inbounds ([1 x i64]* @.c8b, i32 0, i32 0))
  ret i32 0
}

declare i32 @puts(i8* nocapture) nounwind
declare void @foo(i64* nocapture) nounwind

;;;;;

; 64: (('sh_name', 0x0000004e) # '.rodata.str1.1'
; 64-NEXT:   ('sh_type', 0x00000001)
; 64-NEXT:   ('sh_flags', 0x0000000000000032)
; 64-NEXT:   ('sh_addr',
; 64-NEXT:   ('sh_offset',
; 64-NEXT:   ('sh_size', 0x000000000000000d)
; 64-NEXT:   ('sh_link',
; 64-NEXT:   ('sh_info',
; 64-NEXT:   ('sh_addralign', 0x0000000000000001)
; 64-NEXT:   ('sh_entsize', 0x0000000000000001)

; 64: (('sh_name', 0x00000041) # '.rodata.cst8'
; 64-NEXT:   ('sh_type', 0x00000001)
; 64-NEXT:   ('sh_flags', 0x0000000000000012)
; 64-NEXT:   ('sh_addr',
; 64-NEXT:   ('sh_offset',
; 64-NEXT:   ('sh_size', 0x0000000000000010)
; 64-NEXT:   ('sh_link',
; 64-NEXT:   ('sh_info',
; 64-NEXT:   ('sh_addralign', 0x0000000000000008)
; 64-NEXT:   ('sh_entsize', 0x0000000000000008)

