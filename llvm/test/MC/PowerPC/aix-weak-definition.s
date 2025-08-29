## Check that weak_definition symbols are emitted as weak externals.
# Note: On AIX, .weak_definition is mapped to .weak by LLVM's backend.
# RUN: llvm-mc -triple powerpc-ibm-aix-xcoff %s -filetype=obj -o - | \
# RUN:   llvm-objdump --syms - | FileCheck %s

        .weak_definition foo  # LLVM IR WeakDefinition → .weak on AIX
foo:
        blr

# CHECK:      SYMBOL TABLE:
# CHECK-NEXT: 00000000      df *DEBUG*    00000000 .file
# CHECK-NEXT: 00000000 l       .text    00000004
# CHECK-NEXT: 00000000 w     F .text (csect: )  00000000 foo
