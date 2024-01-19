;; Ensure that the .llvm.lto section has SHT_EXCLUDE set.
; RUN: opt --mtriple x86_64-unknown-linux-gnu < %s -passes="embed-bitcode" -S \
; RUN:   | llc --mtriple x86_64-unknown-linux-gnu -filetype=obj \
; RUN:   | llvm-readelf - --sections \
; RUN:   | FileCheck %s --check-prefix=EXCLUDE

; EXCLUDE: Name               Type     {{.*}} ES Flg Lk Inf Al
; EXCLUDE: .llvm.lto          PROGBITS {{.*}} 00   E  0   0  1

@a = global i32 1
