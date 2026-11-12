# REQUIRES: x86-registered-target

# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o -g

.type _Z1cv,@function
_Z1cv:
    nop

# Check the default is to demangle and that the last of --demangle and
# --no-demangle wins.
# RUN: llvm-symbolizer --obj %t.o 0 \
# RUN:    | FileCheck %s --check-prefix=DEMANGLED_FUNCTION_NAME
# RUN: llvm-symbolizer --demangle --obj %t.o 0 \
# RUN:    | FileCheck %s --check-prefix=DEMANGLED_FUNCTION_NAME
# RUN: llvm-symbolizer -C --obj %t.o 0 \
# RUN:    | FileCheck %s --check-prefix=DEMANGLED_FUNCTION_NAME
# RUN: llvm-symbolizer --no-demangle --obj %t.o 0 \
# RUN:    | FileCheck %s --check-prefix=MANGLED_FUNCTION_NAME
# RUN: llvm-symbolizer --demangle --no-demangle --obj %t.o 0 \
# RUN:    | FileCheck %s --check-prefix=MANGLED_FUNCTION_NAME
# RUN: llvm-symbolizer -C --no-demangle --obj %t.o 0 \
# RUN:    | FileCheck %s --check-prefix=MANGLED_FUNCTION_NAME
# RUN: llvm-symbolizer --no-demangle --demangle --obj %t.o 0 \
# RUN:    | FileCheck %s --check-prefix=DEMANGLED_FUNCTION_NAME
# RUN: llvm-symbolizer --no-demangle -C --obj %t.o 0 \
# RUN:    | FileCheck %s --check-prefix=DEMANGLED_FUNCTION_NAME

# Check that for llvm-addr2line the default is not to demangle.
# RUN: llvm-addr2line -fe %t.o 0 \
# RUN:    | FileCheck %s --check-prefix=MANGLED_FUNCTION_NAME
# RUN: llvm-addr2line -fCe %t.o 0 \
# RUN:    | FileCheck %s --check-prefix=DEMANGLED_FUNCTION_NAME

# pprof passes -demangle=false
# RUN: llvm-symbolizer -demangle=false --obj %t.o 0 \
# RUN:    | FileCheck %s --check-prefix=MANGLED_FUNCTION_NAME
# RUN: llvm-symbolizer -demangle=true --obj %t.o 0 \
# RUN:    | FileCheck %s --check-prefix=DEMANGLED_FUNCTION_NAME

# MANGLED_FUNCTION_NAME: _Z1cv
# DEMANGLED_FUNCTION_NAME: c()
