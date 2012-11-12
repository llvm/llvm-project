; RUN: llc  -march=mipsel -mcpu=mips16 -relocation-model=pic -O3 < %s | FileCheck %s -check-prefix=PIC16

@a = thread_local global i32 4, align 4

define i32 @foo() nounwind readonly {
entry:
  %0 = load i32* @a, align 4
; PIC16:	lw	${{[0-9]+}}, %call16(__tls_get_addr)(${{[0-9]+}})
; PIC16:	addiu	${{[0-9]+}}, %tlsgd(a)
  ret i32 %0
}


