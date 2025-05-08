// RUN: llvm-mc -triple=x86_64 -filetype=obj %s | llvm-objdump -tdr - | FileCheck %s

// MC allows ?'s in symbol names as an extension.

// CHECK-LABEL:SYMBOL TABLE:
// CHECK-NEXT: 0000000000000001 l     F .text  0000000000000000 a"b\{{$}}
// CHECK-NEXT: 0000000000000006 l       .text  0000000000000000 a\{{$}}
// CHECK-NEXT: 0000000000000000 g     F .text  0000000000000000 foo?bar
// CHECK-NEXT: 0000000000000000 *UND*          0000000000000000 a"b\q{{$}}
// CHECK-EMPTY:

.text
.globl foo?bar
.type foo?bar, @function
foo?bar:
ret

// CHECK-LABEL:<a"b\>:
// CHECK-NEXT:   callq  {{.*}} <a"b\>
// CHECK-LABEL:<a\>:
// CHECK-NEXT:   callq  {{.*}}
// CHECK-NEXT:     R_X86_64_PLT32 a"b\q-0x4
.type "a\"b\\", @function
"a\"b\\":
  call "a\"b\\"
"a\\":
/// GAS emits a warning for \q
  call "a\"b\q"
