# Check that .weak_definition marks symbols as weak externals on AIX/XCOFF.
#
# .weak_definition is not a documented AIX directive; LLVM emits the
# documented .weak directive instead.
#
# RUN: llvm-mc -triple powerpc-ibm-aix-xcoff %s -filetype=obj -o - | \
# RUN:   llvm-objdump --syms - | FileCheck %s
# RUN: llvm-mc -triple powerpc-ibm-aix-xcoff %s -filetype=asm -o - | \
# RUN:   FileCheck %s --check-prefix=ASM

        .weak_definition foo
foo:
        blr

# CHECK:      SYMBOL TABLE:
# CHECK-NEXT: 00000000      df *DEBUG*    00000000 .file
# CHECK-NEXT: 00000000 l       .text    00000004
# CHECK-NEXT: 00000000 w     F .text (csect: )  00000000 foo

# ASM:        .weak  foo
