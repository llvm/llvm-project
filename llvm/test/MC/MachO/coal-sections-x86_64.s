// RUN: llvm-mc -triple x86_64-apple-darwin -filetype=obj %s -o - 2>%t.err | llvm-readobj --sections | FileCheck %s
// RUN: FileCheck --check-prefix=WARNING < %t.err %s

// CHECK: Section {
// CHECK-NEXT: Index: 0
// CHECK-NEXT: Name: __text (

// CHECK: Section {
// CHECK-NEXT: Index: 1
// CHECK-NEXT: Name: __textcoal_nt (

// CHECK: Section {
// CHECK-NEXT: Index: 2
// CHECK-NEXT: Name: __const_coal (

// CHECK: Section {
// CHECK-NEXT: Index: 3
// CHECK-NEXT: Name: __datacoal_nt (

// WARNING: warning: section "__textcoal_nt" is deprecated
// WARNING: note: change section name to "__text"
// WARNING: warning: section "__const_coal" is deprecated
// WARNING: note: change section name to "__const"
// WARNING: warning: section "__datacoal_nt" is deprecated
// WARNING: note: change section name to "__data"

	.section	__TEXT,__textcoal_nt,coalesced,pure_instructions
	.globl	_foo
	.weak_definition	_foo
	.align	4, 0x90
_foo:
	retq

	.section	__TEXT,__const_coal,coalesced
	.globl	_a                      ## @a
	.weak_definition	_a
	.align	4
_a:
	.long	1                       ## 0x1

	.section	__DATA,__datacoal_nt,coalesced
	.globl	_b                      ## @b
	.weak_definition	_b
	.align	2
_b:
	.long	5                       ## 0x5

.subsections_via_symbols
