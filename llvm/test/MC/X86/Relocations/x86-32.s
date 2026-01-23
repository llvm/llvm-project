# RUN: llvm-mc -filetype=obj -triple=i686 %s -o %t
# RUN: llvm-objdump -dr --no-addresses %t | sed 's/#.*//;/^ *$/d' | FileCheck %s

# CHECK:       e9 05 00 00 00                jmp     {{.*}} <foo>
# CHECK-NEXT:  e9 00 00 00 00                jmp     {{.*}} <foo>
# CHECK-LABEL: <foo>:
# CHECK-NEXT:  0f 84 fa ff ff ff             je      {{.*}} <foo>
# CHECK-NEXT:  0f 84 f4 ff ff ff             je      {{.*}} <foo>
{disp32} jmp foo
jmp.d32 foo
foo:
{disp32} je foo
je.d32 foo

# CHECK-NEXT: c1 0b 0a                               rorl    $0xa, (%ebx)
# CHECK-NEXT:                R_386_8 .text
                rorl    $foo, (%ebx)
