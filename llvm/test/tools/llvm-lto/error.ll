; RUN: not llvm-lto foobar 2>&1 | FileCheck %s
; CHECK: llvm-lto: error loading file 'foobar': {{N|n}}o such file or directory

; RUN: not llvm-lto --list-symbols-only %S/Inputs/empty.bc 2>&1 | FileCheck %s --check-prefix=CHECK-LIST
; CHECK-LIST: llvm-lto: error loading file '{{.*}}/Inputs/empty.bc': The file was not recognized as a valid object file

; RUN: not llvm-lto --list-dependent-libraries-only %S/Inputs/empty.bc 2>&1 | FileCheck %s --check-prefix=CHECK-LIBS
; CHECK-LIBS: llvm-lto: {{.*}}/Inputs/empty.bc: Could not read LTO input file: The file was not recognized as a valid object file

; RUN: not llvm-lto --thinlto %S/Inputs/empty.bc 2>&1 | FileCheck %s --check-prefix=CHECK-THIN
; CHECK-THIN: llvm-lto: error loading file '{{.*}}/Inputs/empty.bc': Invalid bitcode signature
