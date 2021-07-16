# Check section containing code and data with permission executable for the section.
@ RUN: llvm-mc -triple armv7-none-linux -filetype=obj -o %t.o %p/Inputs/1.s
@ RUN: llvm-readelf -s %t.o | FileCheck %s

# Check section containing code and data with no permissions for the section.
@ RUN: llvm-mc -triple armv7-none-linux -filetype=obj -o %t.o %p/Inputs/2.s
@ RUN: llvm-readelf -s %t.o | FileCheck %s

# Check section containing code and data with read/write permissions for the section.
@ RUN: llvm-mc -triple armv7-none-linux -filetype=obj -o %t.o %p/Inputs/3.s
@ RUN: llvm-readelf -s %t.o | FileCheck %s

# Check section containing data with no permissions for the section.
@ RUN: llvm-mc -triple armv7-none-linux -filetype=obj -o %t.o %p/Inputs/4.s
@ RUN: llvm-readelf -s %t.o | FileCheck %s -check-prefix=MAPPINGSYMBOLS

# Check section containing only data with read/write permissions for the section.
@ RUN: llvm-mc -triple armv7-none-linux -filetype=obj -o %t.o %p/Inputs/5.s
@ RUN: llvm-readelf -s %t.o | FileCheck %s -check-prefix=MAPPINGSYMBOLS

# Check section containing the ident string with no permissions for the section.
@ RUN: llvm-mc -triple armv7-none-linux -filetype=obj -o %t.o %p/Inputs/ident.s
@ RUN: llvm-readelf -s %t.o | FileCheck %s -check-prefix=MAPPINGSYMBOLS

# Check section containing the attributes with no permissions for the section.
@ RUN: llvm-mc -triple armv7-none-linux -filetype=obj -o %t.o %p/Inputs/attr.s
@ RUN: llvm-readelf -s %t.o | FileCheck %s -check-prefix=MAPPINGSYMBOLS

# Check section containing code and data with no permissions for the section.
# data comes before code.
@ RUN: llvm-mc -triple armv7-none-linux -filetype=obj -o %t.o %p/Inputs/6.s
@ RUN: llvm-readelf -s %t.o | FileCheck %s -check-prefix=MIX

# Check section containing code and data with no permissions for the section.
# data comes before code.
@ RUN: llvm-mc -triple armv7-none-linux -filetype=obj -o %t.o %p/Inputs/7.s
@ RUN: llvm-readelf -s %t.o | FileCheck %s

#CHECK-DAG: $a
#CHECK-DAG: $d

#MIX: $d
#MIX: $a
#MIX: $d
#MIX: $a

#MAPPINGSYMBOLS-NOT: $a
#MAPPINGSYMBOLS-NOT: $d
