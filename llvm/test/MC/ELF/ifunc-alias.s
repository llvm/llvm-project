// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readelf -s - | FileCheck %s
.text

.type  foo_impl,@function
foo_impl:
  ret

.type  foo_resolver,@function
foo_resolver:
  mov $foo_impl, %rax
  ret

.type  foo,@gnu_indirect_function
.set   foo,foo_resolver

// All things below should be IFunc identical to 'foo'
.set   foo2,foo
.set   foo3,foo2
.type  foo4,@function
.set   foo4,foo3

// But tls_object should not be IFunc
.set   tls,foo
.type  tls,@tls_object

// CHECK: IFUNC   LOCAL  DEFAULT    2 foo
// CHECK: IFUNC   LOCAL  DEFAULT    2 foo2
// CHECK: IFUNC   LOCAL  DEFAULT    2 foo3
// CHECK: IFUNC   LOCAL  DEFAULT    2 foo4
// CHECK: FUNC    LOCAL  DEFAULT    2 foo_impl
// CHECK: FUNC    LOCAL  DEFAULT    2 foo_resolver
// CHECK: TLS     LOCAL  DEFAULT    2 tls
