; RUN: llc --mtriple=loongarch32 --filetype=obj %s -o %t-la32
; RUN: llvm-readelf -h %t-la32 | FileCheck %s --check-prefixes=ILP32,ABI-D --match-full-lines
; RUN: llc --mtriple=loongarch64 --filetype=obj %s -o %t-la64
; RUN: llvm-readelf -h %t-la64 | FileCheck %s --check-prefixes=LP64,ABI-D --match-full-lines

;; Note that we have not support the -target-abi option to select specific ABI.
;; See comments in LoongArchELFStreamer.cpp. So here we only check the default behaviour.
;; After -target-abi is supported, we can add more tests.

; LP64: Class: ELF64
; ILP32: Class: ELF32

; ABI-D: Flags: 0x43, DOUBLE-FLOAT, OBJ-v1

define void @foo() {
  ret void
}
