// RUN: not llvm-mc -triple x86_64-unknown-unknown -x86-asm-syntax=intel %s 2> %t.err
// RUN: FileCheck --check-prefix=CHECK-STDERR < %t.err %s

_test:
// CHECK-LABEL: _test:
// CHECK: xorl    %eax, %eax

	xor	EAX, EAX
	ret

.set  number, 8
.global _foo

.text
  .global main
main:

// CHECK-STDERR:  error: unknown token in expression
  lea RDX, [RAX * number + RBX + _foo]

// CHECK-STDERR:  error: unknown token in expression
  lea RDX, [_foo + RAX * number + RBX]

// CHECK-STDERR:  error: unknown token in expression
  lea RDX, [number + RAX * number + RCX]

// CHECK-STDERR:  error: unknown token in expression
  lea RDX, [_foo + RAX * number]

// CHECK-STDERR:  error: unknown token in expression
  lea RDX, [_foo + RAX * number + RBX]

// CHECK-STDERR: scale factor in address must be 1, 2, 4 or 8
  lea RDX, [number * RAX + RBX + _foo]

// CHECK-STDERR: scale factor in address must be 1, 2, 4 or 8
  lea RDX, [_foo + number * RAX + RBX]

// CHECK-STDERR: scale factor in address must be 1, 2, 4 or 8
  lea RDX, [8 + number * RAX + RCX]

// CHECK-STDERR: scale factor in address must be 1, 2, 4 or 8
lea RDX, [unknown_number * RAX + RBX + _foo]

// CHECK-STDERR: error: BaseReg/IndexReg already set!
lea RDX, [4 * RAX + 27 * RBX + _pat]
