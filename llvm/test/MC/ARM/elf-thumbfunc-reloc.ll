; RUN: llc %s -mtriple=thumbv7-linux-gnueabi -relocation-model=pic \
; RUN: -filetype=obj -o - | elf-dump --dump-section-data | \
; RUN: FileCheck %s

; FIXME: This file needs to be in .s form!
; We wanna test relocatable thumb function call,
; but ARMAsmParser cannot handle "bl foo(PLT)" yet

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:64:128-a0:0:32-n32"
target triple = "thumbv7-none--gnueabi"

define void @foo() nounwind {
entry:
  ret void
}

define void @bar() nounwind {
entry:
  call void @foo()
  ret void
}


; make sure that bl 0 <foo> (fff7feff) is correctly encoded
; CHECK: '_section_data', '704700bf 2de90048 fff7feff bde80088'

;  Offset     Info    Type            Sym.Value  Sym. Name
; 00000008  0000070a R_ARM_THM_CALL    00000001   foo
; CHECK:           Relocation 0
; CHECK-NEXT:      'r_offset', 0x00000008
; CHECK-NEXT:      'r_sym', 0x000007
; CHECK-NEXT:      'r_type', 0x0a

; make sure foo is thumb function: bit 0 = 1
; CHECK:           Symbol 7
; CHECK-NEXT:      'foo'
; CHECK-NEXT:      'st_value', 0x00000001
