# RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu < %s | llvm-readobj --symbols - | FileCheck %s
# RUN: not llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu --defsym ERR=1 %s 2>&1 |\
# RUN:         FileCheck %s --check-prefix=ERR

.byte 0

.lto_set_conditional b, a
.lto_set_conditional d, a
.lto_set_conditional c, b
.lto_set_conditional e, n

# CHECK:      Symbol {
# CHECK:        Name: a
# CHECK-NEXT:   Value: 0x1
a:
.byte 0

# Verify that pending conditional symbols are emitted next

# CHECK:      Symbol {
# CHECK-NEXT:   Name: b
# CHECK-NEXT:   Value: 0x1
# CHECK:      Symbol {
# CHECK-NEXT:   Name: c
# CHECK-NEXT:   Value: 0x1
# CHECK:      Symbol {
# CHECK-NEXT:   Name: d
# CHECK-NEXT:   Value: 0x1

# CHECK-NOT:    Name: e

# Remaining conditional symbols are emitted immediately

# CHECK:      Symbol {
# CHECK-NEXT:   Name: f
# CHECK-NEXT:   Value: 0x1
.lto_set_conditional f, a

# CHECK:      Symbol {
# CHECK-NEXT:   Name: g
# CHECK-NEXT:   Value: 0x1
.lto_set_conditional g, b

# CHECK-NOT:   Name: h
.lto_set_conditional h, m

.ifdef ERR
.text
# ERR: {{.*}}.s:[[#@LINE+1]]:25: error: expected identifier
.lto_set_conditional i, ERR
.endif
