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

