; RUN: llc -filetype=obj -mtriple x86_64-pc-linux-gnu %s -o - | llvm-readobj -S - | FileCheck -check-prefix=64 %s

; Test that constant mergeable strings have sh_entsize set.

@.str1 = private unnamed_addr constant [6 x i8] c"tring\00"
@.str2 = private unnamed_addr constant [7 x i8] c"String\00"
@.c8a = private unnamed_addr constant [1 x i64] [i64 42]
@.c8b = private unnamed_addr constant [1 x i64] [i64 42]

define i32 @main() nounwind {
  %1 = call i32 @puts(i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str1, i32 0, i32 0))
  %2 = call i32 @puts(i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str2, i32 0, i32 0))
  call void @foo(i64* getelementptr inbounds ([1 x i64], [1 x i64]* @.c8a, i32 0, i32 0))
  call void @foo(i64* getelementptr inbounds ([1 x i64], [1 x i64]* @.c8b, i32 0, i32 0))
  ret i32 0
}

declare i32 @puts(i8* nocapture) nounwind
declare void @foo(i64* nocapture) nounwind

;;;;;

; 64:        Section {
; 64:          Name: .rodata.str1.1
; 64-NEXT:     Type: SHT_PROGBITS
; 64-NEXT:     Flags [
; 64-NEXT:       SHF_ALLOC
; 64-NEXT:       SHF_MERGE
; 64-NEXT:       SHF_STRINGS
; 64-NEXT:     ]
; 64-NEXT:     Address:
; 64-NEXT:     Offset:
; 64-NEXT:     Size: 13
; 64-NEXT:     Link:
; 64-NEXT:     Info:
; 64-NEXT:     AddressAlignment: 1
; 64-NEXT:     EntrySize: 1
; 64-NEXT:   }

; 64:        Section {
; 64:          Name: .rodata.cst8
; 64-NEXT:     Type: SHT_PROGBITS
; 64-NEXT:     Flags [
; 64-NEXT:       SHF_ALLOC
; 64-NEXT:       SHF_MERGE
; 64-NEXT:     ]
; 64-NEXT:     Address:
; 64-NEXT:     Offset:
; 64-NEXT:     Size: 16
; 64-NEXT:     Link:
; 64-NEXT:     Info:
; 64-NEXT:     AddressAlignment: 8
; 64-NEXT:     EntrySize: 8
; 64-NEXT:   }
