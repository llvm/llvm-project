# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux-gnu %s | llvm-readelf -hs - | FileCheck %s --check-prefixes=CHECK,GNU
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-freebsd %s | llvm-readelf -hs - | FileCheck %s --check-prefixes=CHECK,FREEBSD

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
# GNU:        OS/ABI: UNIX - GNU
# FREEBSD:    OS/ABI: UNIX - FreeBSD

# CHECK:      FUNC    LOCAL  DEFAULT    2 foo_impl
# CHECK-NEXT: FUNC    LOCAL  DEFAULT    2 foo_resolver
# CHECK-NEXT: IFUNC   LOCAL  DEFAULT    2 foo
