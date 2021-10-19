// Check that ".lto_discard" ignores symbol assignments and attribute changes
// for the specified symbols.
// RUN: llvm-mc -triple x86_64-pc-linux-gnu < %s | FileCheck %s

// Check that ".lto_discard" only accepts identifiers.
// RUN: not llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu --defsym ERR=1 %s 2>&1 |\
// RUN:         FileCheck %s --check-prefix=ERR

// CHECK-NOT:   .weak foo
// CHECK-NOT:       foo:
// CHECK:       .weak bar
// CHECK:           bar:
// CHECK:               .byte 2

.lto_discard foo
.weak foo
foo:
    .byte 1

.lto_discard
.weak bar
bar:
    .byte 2


.ifdef ERR
.text
# ERR: {{.*}}.s:[[#@LINE+1]]:14: error: expected identifier
.lto_discard 1
.endif
