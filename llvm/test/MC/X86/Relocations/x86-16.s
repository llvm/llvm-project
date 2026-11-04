# RUN: llvm-mc -filetype=obj -triple=i386-unknown-code16 %s -o %t
# RUN: llvm-objdump -dr --triple=i386-unknown-code16 %t | FileCheck %s

# CHECK:       e9 03 00                      jmp     {{.*}} <foo>
# CHECK-NEXT:  e9 00 00                      jmp     {{.*}} <foo>
# CHECK-LABEL: <foo>:
# CHECK-NEXT:  0f 84 fc ff                   je      {{.*}} <foo>
# CHECK-NEXT:  0f 84 f8 ff                   je      {{.*}} <foo>
{disp32} jmp foo
jmp.d32 foo
foo:
{disp32} je foo
je.d32 foo
