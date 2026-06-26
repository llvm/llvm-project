// Check that ".lto_discard" ignores symbol assignments and attribute changes
// for the specified symbols.
// RUN: llvm-mc -triple x86_64-pc-linux-gnu < %s | FileCheck %s

// Also check directly via the integrated assembler that a symbol defined
// before ".lto_discard SYM" survives into the object file even when
// ".lto_discard SYM" is followed by another set of attribute/assignment
// directives for SYM. The latter pattern is what the LLVM LTO library
// emits when multiple bitcode inputs each carry the same weak symbol
// definition through file-scope inline asm.
// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o %t.o
// RUN: llvm-readelf --syms %t.o | FileCheck %s --check-prefix=OBJ

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

// "Preserve first, discard later" pattern: ".weak baz; .set baz, V" is
// followed by ".lto_discard baz; .weak baz; .set baz, V". The trailing
// pair must be ignored without disturbing the preserved first definition.
// OBJ:      WEAK   {{.*}} ABS    baz
.weak baz
.set baz, 0xCAFEC0DE
.lto_discard baz
.weak baz
.set baz, 0xCAFEC0DE

// A ".set" that follows a ".lto_discard SYM" for a previously-undefined
// SYM must not give SYM a value; SYM should stay an undef weak reference.
// OBJ:      WEAK   {{.*}} UND    qux
.weak qux
.lto_discard qux
.set qux, 0xDEADBEEF


.ifdef ERR
.text
# ERR: {{.*}}.s:[[#@LINE+1]]:14: error: expected identifier
.lto_discard 1
.endif
