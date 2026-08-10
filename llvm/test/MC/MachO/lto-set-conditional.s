# RUN: llvm-mc -filetype=obj -triple i386-apple-darwin9 < %s | llvm-readobj --symbols - | FileCheck %s
# RUN: not llvm-mc -filetype=obj -triple i386-apple-darwin9 --defsym ERR=1 %s 2>&1 |\
# RUN:         FileCheck %s --check-prefix=ERR

.byte 0

.lto_set_conditional b, a
.lto_set_conditional d, a
.lto_set_conditional c, b
.lto_set_conditional e, n

# CHECK:      Symbol {
# CHECK:        Name: a
# CHECK:        Flags [
# CHECK-NEXT:     NoDeadStrip
# CHECK:        Value: 0x1
a:
.byte 0
.no_dead_strip a

# Verify that pending conditional symbols are emitted next

# CHECK:      Symbol {
# CHECK-NEXT:   Name: b
# CHECK:        Flags [
# CHECK-NEXT:     NoDeadStrip
# CHECK:        Value: 0x1
# CHECK:      Symbol {
# CHECK-NEXT:   Name: c
# CHECK:        Flags [
# CHECK-NEXT:     NoDeadStrip
# CHECK:        Value: 0x1
# CHECK:      Symbol {
# CHECK-NEXT:   Name: d
# CHECK:        Flags [
# CHECK-NEXT:     NoDeadStrip
# CHECK:        Value: 0x1

# CHECK-NOT:    Name: e

# Remaining conditional symbols are emitted immediately

# CHECK:      Symbol {
# CHECK-NEXT:   Name: f
# CHECK:        Flags [
# CHECK-NEXT:     NoDeadStrip
# CHECK:        Value: 0x1
.lto_set_conditional f, a

# CHECK:      Symbol {
# CHECK-NEXT:   Name: g
# CHECK:        Flags [
# CHECK-NEXT:     NoDeadStrip
# CHECK:        Value: 0x1
.lto_set_conditional g, b

# CHECK:      Symbol {
# CHECK-NEXT:   Name: m
# CHECK:        Flags [
# CHECK-NOT :     NoDeadStrip
# CHECK:        Value: 0x2
m:

# CHECK:      Symbol {
# CHECK-NEXT:   Name: h
# CHECK:        Flags [
# CHECK-NOT :     NoDeadStrip
# CHECK:        Value: 0x2
.lto_set_conditional h, m

.ifdef ERR
.text
# ERR: {{.*}}.s:[[#@LINE+1]]:25: error: expected identifier
.lto_set_conditional i, ERR
.endif
