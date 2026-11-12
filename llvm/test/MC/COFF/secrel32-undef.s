# RUN: llvm-mc -filetype=obj -triple i686-pc-win32 %s -o %t.obj
# RUN: llvm-readobj --symbols -r %t.obj | FileCheck %s

# Previously .secrel32 and .secidx relocations against undefined symbols
# resulted in an error. That was a mistake. The linker is fully capable of
# resolving these relocations against symbols in other object files. Such
# relocations can be found in the MSVCRT debug info describing linker-provided
# symbols like __safe_se_handler_table and __guard_fids_table.

.data
foo:
        .secrel32 bar
        .secidx baz


# CHECK: Relocations [
# CHECK:   Section (2) .data {
# CHECK:     0x0 IMAGE_REL_I386_SECREL bar
# CHECK:     0x4 IMAGE_REL_I386_SECTION baz
# CHECK:   }
# CHECK: ]

# CHECK:   Symbol {
# CHECK:     Name: bar
# CHECK-NEXT:     Value: 0
# CHECK-NEXT:     Section: IMAGE_SYM_UNDEFINED (0)
# CHECK:   Symbol {
# CHECK:     Name: baz
# CHECK-NEXT:     Value: 0
# CHECK-NEXT:     Section: IMAGE_SYM_UNDEFINED (0)
