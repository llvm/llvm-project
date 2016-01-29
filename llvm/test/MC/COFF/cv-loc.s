# RUN: llvm-mc < %s -triple=x86_64-pc-win32 -filetype=obj | llvm-readobj - -codeview | FileCheck %s

.section .debug$S
.long 4
.cv_stringtable

.cv_file 1 "a.c"
.cv_file 2 "t.inc"

# Implements this C:
# void f(volatile int *x) {
#   ++*x;
# #include "t.h" // contains two ++*x; statements
#   ++*x;
# }

.text
.def     f;
        .scl    2;
        .type   32;
        .endef
        .text
        .globl  f
        .align  16, 0x90
f:
.Lfunc_begin0:
  .cv_loc 0 1 5 2
  incl (%rdi)
  # #include "t.h" start
  .cv_loc 0 2 0 0
  incl (%rdi)
  .cv_loc 0 2 1 0
  incl (%rdi)
  # #include "t.h" end
  .cv_loc 0 1 6 2
  incl (%rdi)
  retq
.Lfunc_end0:

.section .debug$S
.cv_filechecksums
.cv_linetable 0, f, .Lfunc_end0

# CHECK: FunctionLineTable [
# CHECK:   LinkageName: f
# CHECK:   Flags: 0x1
# CHECK:   CodeSize: 0x9
# CHECK:   FilenameSegment [
# CHECK:     Filename: a.c (0x0)
# CHECK:     +0x0 [
# CHECK:       LineNumberStart: 5
# CHECK:       LineNumberEndDelta: 0
# CHECK:       IsStatement: Yes
# CHECK:       ColStart: 2
# CHECK:       ColEnd: 0
# CHECK:     ]
# CHECK:   ]
# CHECK:   FilenameSegment [
# CHECK:     Filename: t.inc (0x8)
# CHECK:     +0x2 [
# CHECK:       LineNumberStart: 0
# CHECK:       LineNumberEndDelta: 0
# CHECK:       IsStatement: Yes
# CHECK:       ColStart: 0
# CHECK:       ColEnd: 0
# CHECK:     ]
# CHECK:     +0x4 [
# CHECK:       LineNumberStart: 1
# CHECK:       LineNumberEndDelta: 0
# CHECK:       IsStatement: Yes
# CHECK:       ColStart: 0
# CHECK:       ColEnd: 0
# CHECK:     ]
# CHECK:   ]
# CHECK:   FilenameSegment [
# CHECK:     Filename: a.c (0x0)
# CHECK:     +0x6 [
# CHECK:       LineNumberStart: 6
# CHECK:       LineNumberEndDelta: 0
# CHECK:       IsStatement: Yes
# CHECK:       ColStart: 2
# CHECK:       ColEnd: 0
# CHECK:     ]
# CHECK:   ]
# CHECK: ]
