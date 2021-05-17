# RUN: llvm-mc -filetype=obj -triple=powerpc64-unknown-freebsd13.0 %s -o %t
# RUN: llvm-readelf -s %t | FileCheck %s

# CHECK:      Type Bind   Vis                     Ndx Name
# CHECK:      FUNC GLOBAL DEFAULT [<other: 0x60>]   2 foo
# CHECK-NEXT: FUNC WEAK   DEFAULT [<other: 0x60>]   2 __impl_foo
# CHECK-NEXT: FUNC GLOBAL DEFAULT [<other: 0x60>]   2 func
# CHECK-NEXT: FUNC WEAK   DEFAULT [<other: 0x60>]   2 weak_func
# CHECK:      FUNC WEAK   DEFAULT [<other: 0x60>]   2 foo@FBSD_1.1

.text
.abiversion 2

.globl foo
.type foo,@function
foo:
  nop
  nop
  .localentry foo, 8

.symver __impl_foo, foo@FBSD_1.1
.weak   __impl_foo
.set    __impl_foo, foo

.globl  func
# Mimick FreeBSD weak function/reference
.weak   weak_func
.equ    weak_func, func

.p2align 2
.type    func,@function
func:
  nop
  nop
  .localentry func, 8

## PR44284 Don't crash if err is redefined after .set
.set err, _err
.globl err
err:
