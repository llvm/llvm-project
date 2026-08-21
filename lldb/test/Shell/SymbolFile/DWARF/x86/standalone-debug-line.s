# Test malformed and unusual standalone DWARF line tables. The ordinary
# address and source lookup behavior is covered by the SBAPI test in
# lldb/test/API/python_api/symbol-context/standalone-debug-line.

# REQUIRES: lld
# UNSUPPORTED: system-windows

# RUN: llvm-mc -triple=x86_64-pc-linux -filetype=obj -dwarf-version=2 \
# RUN:   %s -o %t.good.o
# RUN: llvm-mc -triple=x86_64-pc-linux -filetype=obj \
# RUN:   --defsym MALFORMED_BODY=1 %s -o %t.bad-body.o
# RUN: ld.lld -e bad_body -Ttext=0x201000 %t.bad-body.o -o %t.bad-body
# RUN: %lldb %t.bad-body -b -o "image dump symfile" \
# RUN:   -o "image lookup -a 0x201000 -v" | \
# RUN:   FileCheck %s --check-prefix=BAD-BODY
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux \
# RUN:   %S/debug-types-line-tables.s -o %t.types.o
# RUN: llvm-objcopy --remove-section=.debug_info %t.types.o %t.types
# RUN: lldb-test symbols %t.types | FileCheck %s --check-prefix=TYPES
# RUN: llvm-mc -triple=x86_64-pc-linux -filetype=obj --defsym INFO_ONLY=1 \
# RUN:   %s -o %t.info.o
# RUN: ld.lld -e foo -Ttext=0x201000 %t.good.o %t.info.o -o %t.info
# RUN: %lldb %t.info -b -o "image dump symfile" \
# RUN:   -o "image lookup -a 0x201000 -v" | \
# RUN:   FileCheck %s --check-prefix=INFO
# RUN: llvm-mc -triple=x86_64-pc-linux -filetype=obj --defsym RECOVERABLE=1 \
# RUN:   %s -o %t.recoverable.o
# RUN: ld.lld -e recoverable -Ttext=0x201000 %t.recoverable.o \
# RUN:   -o %t.recoverable
# RUN: %lldb %t.recoverable -b -o "image dump symfile" \
# RUN:   -o "image lookup -a 0x201000 -v" | \
# RUN:   FileCheck %s --check-prefix=RECOVERABLE

        .ifdef INFO_ONLY
        .section .debug_info,"",@progbits
        .byte 0
        .else
        .ifdef MALFORMED_BODY
        .text
        .globl bad_body
        .type bad_body,@function
bad_body:
        nop
        retq
.Lbad_body_end:
        .size bad_body, .Lbad_body_end-bad_body

        .section .debug_line,"",@progbits
.Lbad_body_start:
        .long .Lbad_body_end_table-.Lbad_body_version
.Lbad_body_version:
        .short 2
        .long .Lbad_body_header_end-.Lbad_body_header
.Lbad_body_header:
        .byte 1
        .byte 1
        .byte -5
        .byte 14
        .byte 13
        .byte 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1
        .asciz "/tmp"
        .byte 0
        .asciz "bad-body.c"
        .uleb128 1
        .uleb128 0
        .uleb128 0
        .byte 0
.Lbad_body_header_end:
        .byte 0, 9, 2
        .quad bad_body
        .byte 3
        .sleb128 6
        .byte 1
        .byte 2
        .uleb128 2
        .byte 0, 1, 1
        .byte 3
.Lbad_body_end_table:
        .else
        .ifdef RECOVERABLE
        .text
        .globl recoverable
        .type recoverable,@function
recoverable:
        nop
        retq
.Lrecoverable_func_end:
        .size recoverable, .Lrecoverable_func_end-recoverable

        .section .debug_line,"",@progbits
.Lrecoverable_start:
        .long .Lrecoverable_end-.Lrecoverable_version
.Lrecoverable_version:
        .short 2
        .long .Lrecoverable_header_end-.Lrecoverable_header
.Lrecoverable_header:
        .byte 1
        .byte 1
        .byte -5
        .byte 14
        .byte 13
        .byte 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1
        .asciz "/tmp"
        .byte 0
        .asciz "recoverable.c"
        .uleb128 1
        .uleb128 0
        .uleb128 0
        .byte 0
        # CUDA line tables can contain padding at the end of the prologue.
        .byte 0, 0, 0, 0
.Lrecoverable_header_end:
        .byte 0, 9, 2
        .quad recoverable
        .byte 3
        .sleb128 6
        .byte 5
        .uleb128 5
        .byte 1
        .byte 2
        .uleb128 2
        .byte 0, 1, 1
.Lrecoverable_end:
        .else
        .file 1 "/tmp/standalone-one.c"

        .text
        .p2align 4
        .globl foo
        .type foo,@function
foo:
        .loc 1 42 7
        nop
        .loc 1 43 3
        nop
        retq
.Lfunc_end:
        .size foo, .Lfunc_end-foo
        .endif
        .endif
        .endif

# BAD-BODY: SymbolFile symtab
# BAD-BODY-LABEL: (lldb) image lookup -a 0x201000 -v
# BAD-BODY-NOT: CompileUnit:
# BAD-BODY: Symbol: {{.*}}name="bad_body"

# TYPES: Compile units:
# TYPES: CompileUnit{{.*}}file = '/tmp/b.cc'

# INFO: SymbolFile symtab
# INFO-LABEL: (lldb) image lookup -a 0x201000 -v
# INFO-NOT: CompileUnit:
# INFO: Symbol: {{.*}}name="foo"

# RECOVERABLE: SymbolFile dwarf
# RECOVERABLE-LABEL: (lldb) image lookup -a 0x201000 -v
# RECOVERABLE: CompileUnit: {{.*}}file = "/tmp/recoverable.c"
# RECOVERABLE: LineEntry: {{.*}}/tmp/recoverable.c:7:5
# RECOVERABLE: Symbol: {{.*}}name="recoverable"
