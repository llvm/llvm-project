; If the binary looks up libraries using an rpath, we can't test this
; without copying the whole lib dir or polluting the build dir.
; REQUIRES: static-libs

; The above also applies if the binary is built with libc++.
; UNSUPPORTED: libcxx-used

; RUN: echo > %t.input

; workaround for https://openradar.appspot.com/FB8914243
; RUN: rm -f %t.bin--gisel
; RUN: rm -f %t.bin--gisel-O2
; RUN: rm -f %t.bin--unexist

; RUN: cp llvm-isel-fuzzer %t.bin--gisel
; RUN: not %t.bin--gisel %t.input 2>&1 | FileCheck -check-prefix=GISEL %s
; GISEL: Injected args: -global-isel -O0
; GISEL: -mtriple must be specified

; RUN: cp llvm-isel-fuzzer %t.bin--gisel-O2
; RUN: not %t.bin--gisel-O2 %t.input 2>&1 | FileCheck -check-prefix=GISEL-O2 %s
; GISEL-O2: Injected args: -global-isel -O0 -O2
; GISEL-O2: -mtriple must be specified

; RUN: cp llvm-isel-fuzzer %t.bin--unexist
; RUN: not %t.bin--unexist %t.input 2>&1 | FileCheck -check-prefix=NO-OPT %s
; NO-OPT: Unknown option:
