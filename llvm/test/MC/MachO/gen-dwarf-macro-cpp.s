// RUN: llvm-mc -g -triple i386-apple-darwin10 %s -filetype=obj -o %t
// RUN: llvm-dwarfdump -debug-line %t | FileCheck %s

# 1 "foo.S" 2
.macro switcher
        ljmp *0x38(%ecx)
.endmacro
        switcher NaClSwitchNoSSE, 0

// PR14264 was a crash in the code caused by the .macro not handled correctly
// rdar://12637628

// We check that the source name "foo.S" is picked up
// CHECK:                 Dir  Mod Time   File Len   File Name
// CHECK:                 ---- ---------- ---------- ---------------------------
// CHECK: file_names[  1]    1 0x00000000 0x00000000 gen-dwarf-macro-cpp.s
// CHECK: file_names[  2]    0 0x00000000 0x00000000 foo.S
