# RUN: llvm-mc %s -filetype obj -triple wasm32-unknown-unknown -o %t.o \
# RUN:         -split-dwarf-file=%t.dwo -dwarf-version=5
# RUN: llvm-dwp %t.dwo -o %t.dwp
# RUN: llvm-dwarfdump -v %t.dwp | FileCheck %s

# This test checks whether llvm-dwp is able to emit object files of the same
# triple as its inputs.

# CHECK: file format WASM

# Empty file, we just care about the file type.
