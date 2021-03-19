// Check that ".lto_discard" ignores symbol assignments and attribute changes
// for the specified symbols.
// RUN: llvm-mc -triple x86_64 < %s | FileCheck %s

// Check that ".lto_discard" only accepts identifiers.
// RUN: not llvm-mc -filetype=obj -triple x86_64 --defsym ERR=1 %s 2>&1 |\
// RUN:         FileCheck %s --check-prefix=ERR

// CHECK: .weak foo
// CHECK: foo:
// CHECK:    .byte 1
// CHECK: .weak bar
// CHECK: bar:
// CHECK:    .byte 2

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
# ERR: {{.*}}.s:[[#@LINE+1]]:14: error: expected identifier in directive
.lto_discard 1
.endif
