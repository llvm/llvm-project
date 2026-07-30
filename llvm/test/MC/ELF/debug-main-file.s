// RUN: mkdir -p %t/Inputs
// RUN: cp %s %t/base.s
// RUN: cp %s %t/Inputs/subdir.s
// RUN: cd %t

// RUN: llvm-mc -triple=x86_64-linux-unknown -filetype=obj -dwarf-version=4 \
// RUN:     -g base.s -o %t1.o
// RUN: llvm-dwarfdump -debug-info %t1.o | \
// RUN:     FileCheck %s --check-prefixes=CHECK,BASE
// RUN: llvm-mc -triple=x86_64-linux-unknown -filetype=obj -dwarf-version=4 \
// RUN:     -g base.s -o %t2.o -main-file-name rename.s
// RUN: llvm-dwarfdump -debug-info %t2.o | \
// RUN:     FileCheck %s --check-prefixes=CHECK,RENAME
// RUN: llvm-mc -triple=x86_64-linux-unknown -filetype=obj -dwarf-version=4 \
// RUN:     -g Inputs/subdir.s -o %t3.o
// RUN: llvm-dwarfdump -debug-info %t3.o | \
// RUN:     FileCheck %s --check-prefixes=CHECK,SUBDIR
// RUN: llvm-mc -triple=x86_64-linux-unknown -filetype=obj -dwarf-version=4 \
// RUN:     -g Inputs/subdir.s -main-file-name sub-rename.s -o %t4.o
// RUN: llvm-dwarfdump -debug-info %t4.o | \
// RUN:     FileCheck %s --check-prefixes=CHECK,SUB-RENAME

// CHECK: DW_TAG_compile_unit
// CHECK-NOT: DW_TAG
// CHECK: DW_AT_name
// BASE-SAME:       ("base.s")
// RENAME-SAME:     ("rename.s")
// SUBDIR-SAME:     ("Inputs{{(/|\\)+}}subdir.s")
// SUB-RENAME-SAME: ("Inputs{{(/|\\)+}}sub-rename.s")

// CHECK: DW_TAG_label
// CHECK-NOT: DW_TAG
// CHECK: DW_AT_decl_file
// BASE-SAME:       ("{{.*(/|\\)}}base.s")
// RENAME-SAME:     ("{{.*(/|\\)}}rename.s")
// SUBDIR-SAME:     ("{{.*Inputs(/|\\)+}}subdir.s")
// SUB-RENAME-SAME: ("{{.*Inputs(/|\\)+}}sub-rename.s")

        .text
start:
        nop
