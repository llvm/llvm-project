; DISABLE: llc -filetype=obj -mtriple mips-unknown-linux %s -o - | llvm-readobj -h -S --sd | FileCheck %s
; RUN: false
; XFAIL: *

; Check that this is big endian.
; CHECK: ElfHeader {
; CHECK:   Ident {
; CHECK:     DataEncoding: BigEndian
; CHECK:   }
; CHECK: }

; Make sure that a section table (text) entry is correct.
; CHECK:      Sections [
; CHECK:        Section {
; CHECK:          Index:
; CHECK:          Name: .text
; CHECK-NEXT:     Type: SHT_PROGBITS
; CHECK-NEXT:     Flags [ (0x6)
; CHECK-NEXT:       SHF_ALLOC
; CHECK-NEXT:       SHF_EXECINSTR
; CHECK-NEXT:     ]
; CHECK-NEXT:     Address: 0x{{[0-9,A-F]+}}
; CHECK-NEXT:     Offset: 0x{{[0-9,A-F]+}}
; CHECK-NEXT:     Size: {{[0-9]+}}
; CHECK-NEXT:     Link: 0
; CHECK-NEXT:     Info: 0
; CHECK-NEXT:     AddressAlignment: 4
; CHECK-NEXT:     EntrySize: 0

; See that at least first 3 instructions are correct: GP prologue
; CHECK-NEXT:     SectionData (
; CHECK-NEXT:       0000: 3C1C0000 279C0000 0399E021 {{[0-9,A-F, ]*}}
; CHECK:          )
; CHECK:   }

; ModuleID = '../br1.c'
target datalayout = "E-p:32:32:32-i1:8:8-i8:8:32-i16:16:32-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-n32"
target triple = "mips-unknown-linux"

@x = global i32 1, align 4
@str = private unnamed_addr constant [4 x i8] c"goo\00"
@str2 = private unnamed_addr constant [4 x i8] c"foo\00"

define i32 @main() nounwind {
entry:
  %0 = load i32, i32* @x, align 4
  %tobool = icmp eq i32 %0, 0
  br i1 %tobool, label %if.end, label %foo

if.end:                                           ; preds = %entry
  %puts = tail call i32 @puts(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @str, i32 0, i32 0))
  br label %foo

foo:                                              ; preds = %entry, %if.end
  %puts2 = tail call i32 @puts(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @str2, i32 0, i32 0))
  ret i32 0
}

declare i32 @puts(i8* nocapture) nounwind

