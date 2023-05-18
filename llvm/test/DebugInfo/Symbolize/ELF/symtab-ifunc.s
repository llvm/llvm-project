# REQUIRES: x86-registered-target
## Test we can symbolize STT_GNU_IFUNC symbols.
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t
# RUN: llvm-symbolizer --obj=%t 0 1

# CHECK:       g_ifunc
# CHECK-NEXT:  ??:0:0
# CHECK-EMPTY:
# CHECK-NEXT:  l_ifunc
# CHECK-NEXT:  ??:0:0
# CHECK-EMPTY:

## TODO Find the preceding STT_FILE symbol as the filename of l_ifunc.
.file "symtab-ifunc.s"

.Lg_resolver:
  ret
.size .Lg_resolver, 1

.globl g_ifunc
.set g_ifunc, .Lg_resolver
.type g_ifunc, @gnu_indirect_function

.Ll_resolver:
  ret
.size .Ll_resolver, 1

.set l_ifunc, .Ll_resolver
.type l_ifunc, @gnu_indirect_function
