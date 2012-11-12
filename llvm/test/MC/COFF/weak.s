// This tests that default-null weak symbols (a GNU extension) are created
// properly via the .weak directive.

// RUN: llvm-mc -filetype=obj -triple i686-pc-win32 < %s | coff-dump.py | FileCheck %s

    .def    _main;
    .scl    2;
    .type   32;
    .endef
    .text
    .globl  _main
    .align  16, 0x90
_main:                                  # @main
# BB#0:                                 # %entry
    subl    $4, %esp
    movl    $_test_weak, %eax
    testl   %eax, %eax
    je      LBB0_2
# BB#1:                                 # %if.then
    calll   _test_weak
    movl    $1, %eax
    addl    $4, %esp
    ret
LBB0_2:                                 # %return
    xorl    %eax, %eax
    addl    $4, %esp
    ret

    .weak   _test_weak

// CHECK: Symbols = [

// CHECK:      Name               = _test_weak
// CHECK-NEXT: Value              = 0
// CHECK-NEXT: SectionNumber      = 0
// CHECK-NEXT: SimpleType         = IMAGE_SYM_TYPE_NULL (0)
// CHECK-NEXT: ComplexType        = IMAGE_SYM_DTYPE_NULL (0)
// CHECK-NEXT: StorageClass       = IMAGE_SYM_CLASS_WEAK_EXTERNAL (105)
// CHECK-NEXT: NumberOfAuxSymbols = 1
// CHECK-NEXT: AuxillaryData      =
// CHECK-NEXT: 05 00 00 00 02 00 00 00 - 00 00 00 00 00 00 00 00 |................|
// CHECK-NEXT: 00 00                                             |..|

// CHECK:      Name               = .weak._test_weak.default
// CHECK-NEXT: Value              = 0
// CHECK-NEXT: SectionNumber      = 65535
// CHECK-NEXT: SimpleType         = IMAGE_SYM_TYPE_NULL (0)
// CHECK-NEXT: ComplexType        = IMAGE_SYM_DTYPE_NULL (0)
// CHECK-NEXT: StorageClass       = IMAGE_SYM_CLASS_EXTERNAL (2)
// CHECK-NEXT: NumberOfAuxSymbols = 0
// CHECK-NEXT: AuxillaryData      =
