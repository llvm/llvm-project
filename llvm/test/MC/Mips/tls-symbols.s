# RUN: llvm-mc -arch=mips < %s -position-independent -filetype=obj \
# RUN:   | llvm-readelf -symbols | FileCheck %s
# RUN: llvm-mc -arch=mips < %s -filetype=obj | llvm-readelf -symbols \
# RUN:   | FileCheck %s

# Test that TLS relocations cause symbols to be marked as TLS symbols.

  .set  noat
  lui $3, %tlsgd(foo1)
  lui $1, %dtprel_hi(foo2)
  lui $1, %dtprel_lo(foo3)
  lui $1, %tprel_hi(foo4)
  lui $1, %tprel_lo(foo5)
  lw $2, %gottprel(foo6)($28)

  .hidden foo1
  .hidden foo2
  .hidden foo3
  .hidden foo4
  .hidden foo5
  .hidden foo6

# CHECK:     1: {{.+}}       {{.+}} TLS     GLOBAL HIDDEN   UND foo1
# CHECK:     2: {{.+}}       {{.+}} TLS     GLOBAL HIDDEN   UND foo2
# CHECK:     3: {{.+}}       {{.+}} TLS     GLOBAL HIDDEN   UND foo3
# CHECK:     4: {{.+}}       {{.+}} TLS     GLOBAL HIDDEN   UND foo4
# CHECK:     5: {{.+}}       {{.+}} TLS     GLOBAL HIDDEN   UND foo5
# CHECK:     6: {{.+}}       {{.+}} TLS     GLOBAL HIDDEN   UND foo6
