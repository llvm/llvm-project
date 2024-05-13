// RUN: not llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o /dev/null 2>&1 | FileCheck %s

// Check that we diagnose malformed .cfi_llvm_def_aspace_cfa directives.

.cfi_startproc

// CHECK: [[#@LINE+1]]:{{[0-9]+}}: error: invalid register name
.cfi_llvm_def_aspace_cfa foo

// CHECK: [[#@LINE+1]]:{{[0-9]+}}: error: expected comma
.cfi_llvm_def_aspace_cfa %rcx .

// CHECK: [[#@LINE+1]]:{{[0-9]+}}: error: expected absolute expression
.cfi_llvm_def_aspace_cfa %rcx, .+1

// CHECK: [[#@LINE+1]]:{{[0-9]+}}: error: expected comma
.cfi_llvm_def_aspace_cfa %rcx, 1 .

// CHECK: [[#@LINE+1]]:{{[0-9]+}}: error: expected absolute expression
.cfi_llvm_def_aspace_cfa %rcx, 1, .+1

// CHECK: [[#@LINE+1]]:{{[0-9]+}}: error: expected newline
.cfi_llvm_def_aspace_cfa %rcx, 1, 1,

.cfi_endproc
