; Check how tricky symbols are printed in the asm output.
; RUN: llc -mtriple=i686-pc-win32 %s -o - | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=i686-pc-win32 %s -x86-asm-syntax=intel -o - | FileCheck %s --check-prefix=ASM

; Check that we can roundtrip these names through our assembler,
; in both at&t and intel syntax.
; RUN: llc -mtriple=i686-pc-win32 %s -o - | llvm-mc -triple i686-pc-win32 -filetype=obj | llvm-readobj --symbols - | FileCheck %s --check-prefix=READOBJ
; RUN: llc -mtriple=i686-pc-win32 -x86-asm-syntax=intel %s -o - | llvm-mc -triple i686-pc-win32 -filetype=obj | llvm-readobj --symbols - | FileCheck %s --check-prefix=READOBJ


@"\01??__E_Generic_object@?$_Error_objects@H@std@@YAXXZ" = global i32 0
@"\01__ZL16ExceptionHandlerP19_EXCEPTION_POINTERS@4" = global i32 0
@"\01@foo.bar" = global i32 0

define weak i32 @"\01??_B?$num_put@_WV?$back_insert_iterator@V?$basic_string@_WU?$char_traits@_W@std@@V?$allocator@_W@2@@std@@@std@@@std@@51"() section ".text" {
  %a = load i32, ptr @"\01??__E_Generic_object@?$_Error_objects@H@std@@YAXXZ"
  %b = load i32, ptr @"\01__ZL16ExceptionHandlerP19_EXCEPTION_POINTERS@4"
  %c = load i32, ptr @"\01@foo.bar"
  %x = add i32 %a, %b
  %y = add i32 %x, %c
  ret i32 %y
}

; Check that these symbols are not quoted. They occur in output that gets passed to GAS.
; ASM: .globl __ZL16ExceptionHandlerP19_EXCEPTION_POINTERS@4
; ASM-NOT: .globl "__ZL16ExceptionHandlerP19_EXCEPTION_POINTERS@4"
; ASM: .globl @foo.bar
; ASM-NOT: .globl "@foo.bar"

; READOBJ: Symbol
; READOBJ: Name: .text
; READOBJ: Section: .text
; READOBJ: Symbol
; READOBJ: Name: ??_B?$num_put@_WV?$back_insert_iterator@V?$basic_string@_WU?$char_traits@_W@std@@V?$allocator@_W@2@@std@@@std@@@std@@51
; READOBJ: Section: .text
; READOBJ: Symbol
; READOBJ: Name: ??__E_Generic_object@?$_Error_objects@H@std@@YAXXZ
; READOBJ: Symbol
; READOBJ: Name: __ZL16ExceptionHandlerP19_EXCEPTION_POINTERS@4
; READOBJ: Symbol
; READOBJ: Name: @foo.bar
