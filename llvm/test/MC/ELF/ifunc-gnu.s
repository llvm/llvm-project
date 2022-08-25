# REQUIRES: x86-registered-target
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux-gnu %s -o -| llvm-readelf -hs - | FileCheck %s

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

## ELFOSABI_NONE is changed to ELFOSABI_GNU. Other OSABI values are unchanged.
# CHECK:      OS/ABI: UNIX - GNU
# CHECK:      FUNC    LOCAL  DEFAULT    2 foo_impl
# CHECK-NEXT: FUNC    LOCAL  DEFAULT    2 foo_resolver
# CHECK-NEXT: IFUNC   LOCAL  DEFAULT    2 foo
