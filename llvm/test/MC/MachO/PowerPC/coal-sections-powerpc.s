// RUN: llvm-mc -triple powerpc-apple-darwin8 -arch=ppc32 -filetype=obj %s -o - | llvm-readobj --sections | FileCheck %s

// CHECK: Section {
// CHECK-NEXT: Index: 0

// CHECK: Section {
// CHECK-NEXT: Index: 1
// CHECK-NEXT: Name: __textcoal_nt (

// CHECK: Section {
// CHECK-NEXT: Index: 2

// CHECK: Section {
// CHECK-NEXT: Index: 3
// CHECK-NEXT: Name: __const_coal (

// CHECK: Section {
// CHECK-NEXT: Index: 4
// CHECK-NEXT: Name: __datacoal_nt (

  .section  __TEXT,__text,regular,pure_instructions
  .machine ppc
  .section  __TEXT,__textcoal_nt,coalesced,pure_instructions
  .section  __TEXT,__symbol_stub1,symbol_stubs,pure_instructions,16
  .section  __TEXT,__text,regular,pure_instructions
  .section  __TEXT,__textcoal_nt,coalesced,pure_instructions
  .globl  _foo
  .weak_definition  _foo
  .align  4
_foo:
	blr

.subsections_via_symbols
	.section	__TEXT,__const_coal,coalesced
	.globl	_a                      ; @a
	.weak_definition	_a
	.align	4
_a:
	.long	1                       ; 0x1

	.section	__DATA,__datacoal_nt,coalesced
	.globl	_b                      ; @b
	.weak_definition	_b
	.align	2
_b:
	.long	5                       ; 0x5
