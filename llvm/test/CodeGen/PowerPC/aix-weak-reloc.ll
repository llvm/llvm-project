; RUN: llc -mtriple powerpc-ibm-aix-xcoff -mcpu=ppc -filetype=obj -o %t.o < %s
; RUN: llvm-objdump -dr %t.o | FileCheck --check-prefix=OBJ32 %s

; RUN: llc -mtriple powerpc64-ibm-aix-xcoff -mcpu=ppc -filetype=obj -o %t.o < %s
; RUN: llvm-objdump -dr %t.o | FileCheck --check-prefix=OBJ64 %s

; Function Attrs: noinline nounwind optnone
define i32 @main() {
entry:
  %retval = alloca i32, align 4
  store i32 0, ptr %retval, align 4
  %call = call i32 @foo()
  ret i32 %call
}

; Function Attrs: noinline nounwind optnone
define weak i32 @foo() {
entry:
  ret i32 3
}

; OBJ32:       00000000 <.main>:
; OBJ32-NEXT:         0: 7c 08 02 a6  	mflr 0
; OBJ32-NEXT:         4: 94 21 ff c0  	stwu 1, -64(1)
; OBJ32-NEXT:         8: 38 60 00 00  	li 3, 0
; OBJ32-NEXT:         c: 90 01 00 48  	stw 0, 72(1)
; OBJ32-NEXT:        10: 90 61 00 3c  	stw 3, 60(1)
; OBJ32-NEXT:        14: 48 00 00 31  	bl 0x44 <.foo>
; OBJ32-NEXT:  			00000014:  R_RBR	.foo
; OBJ32-NEXT:        18: 60 00 00 00  	nop
; OBJ32:       00000044 <.foo>:
; OBJ32-NEXT:        44: 38 60 00 03  	li 3, 3
; OBJ32-NEXT:        48: 4e 80 00 20  	blr

; OBJ64:      0000000000000000 <.main>:
; OBJ64-NEXT:        0: 7c 08 02 a6  	mflr 0
; OBJ64-NEXT:        4: f8 21 ff 81  	stdu 1, -128(1)
; OBJ64-NEXT:        8: 38 60 00 00  	li 3, 0
; OBJ64-NEXT:        c: f8 01 00 90  	std 0, 144(1)
; OBJ64-NEXT:       10: 90 61 00 7c  	stw 3, 124(1)
; OBJ64-NEXT:       14: 48 00 00 31  	bl 0x44 <.foo>
; OBJ64-NEXT: 		0000000000000014:  R_RBR	.foo
; OBJ64-NEXT:       18: 60 00 00 00  	nop
; OBJ64:      0000000000000044 <.foo>:
; OBJ64-NEXT:       44: 38 60 00 03  	li 3, 3
; OBJ64-NEXT:       48: 4e 80 00 20  	blr
