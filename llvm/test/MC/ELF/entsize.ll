; RUN: llc -filetype=obj -mtriple x86_64-pc-linux-gnu %s -o - | llvm-readobj -S - | FileCheck -check-prefix=64 %s

; Test that constant mergeable strings have sh_entsize set.

@.str1 = private unnamed_addr constant [6 x i8] c"tring\00"
@.str2 = private unnamed_addr constant [7 x i8] c"String\00"
@.c8a = private unnamed_addr constant [1 x i64] [i64 42]
@.c8b = private unnamed_addr constant [1 x i64] [i64 42]

define i32 @main() nounwind {
  %1 = call i32 @puts(ptr @.str1)
  %2 = call i32 @puts(ptr @.str2)
  call void @foo(ptr @.c8a)
  call void @foo(ptr @.c8b)
  ret i32 0
}

declare i32 @puts(ptr nocapture) nounwind
declare void @foo(ptr nocapture) nounwind

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
