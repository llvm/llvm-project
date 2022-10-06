; RUN: llc --mtriple=loongarch32 --filetype=obj %s -o %t-la32
; RUN: llvm-readelf -h %t-la32 | FileCheck %s --check-prefix=ILP32D --match-full-lines
; RUN: llc --mtriple=loongarch64 --filetype=obj %s -o %t-la64
; RUN: llvm-readelf -h %t-la64 | FileCheck %s --check-prefix=LP64D --match-full-lines

;; Note that we have not support the -target-abi option to select specific ABI.
;; See comments in LoongArchELFStreamer.cpp. So here we only check the default behaviour.
;; After -target-abi is supported, we can add more tests.

; LP64D: Flags: 0x3, LP64, DOUBLE-FLOAT
; ILP32D: Flags: 0x7, ILP32, DOUBLE-FLOAT

define void @foo() {
  ret void
}
