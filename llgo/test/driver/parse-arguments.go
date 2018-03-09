// RUN: not llgo -B 2>&1 | FileCheck --check-prefix=B %s
// RUN: not llgo -D 2>&1 | FileCheck --check-prefix=D %s
// RUN: not llgo -I 2>&1 | FileCheck --check-prefix=I %s
// RUN: not llgo -isystem 2>&1 | FileCheck --check-prefix=isystem %s
// RUN: not llgo -L 2>&1 | FileCheck --check-prefix=L %s
// RUN: not llgo -fload-plugin 2>&1 | FileCheck --check-prefix=fload-plugin %s
// RUN: not llgo -mllvm 2>&1 | FileCheck --check-prefix=mllvm %s
// RUN: not llgo -o 2>&1 | FileCheck --check-prefix=o %s

// B: missing argument after '-B'
// D: missing argument after '-D'
// I: missing argument after '-I'
// isystem: missing argument after '-isystem'
// L: missing argument after '-L'
// fload-plugin: missing argument after '-fload-plugin'
// mllvm: missing argument after '-mllvm'
// o: missing argument after '-o'
