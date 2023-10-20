; RUN: llvm-as < %s | llvm-dis | FileCheck %s --check-prefix=ASSEM-DISASS

define decimal32 @check_decimal32(decimal32 %A) {
; ASSEM-DISASS: ret decimal32 %A
    ret decimal32 %A
}

define decimal64 @check_decimal64(decimal64 %A) {
; ASSEM-DISASS: ret decimal64 %A
    ret decimal64 %A
}

define decimal128 @check_decimal128(decimal128 %A) {
; ASSEM-DISASS: ret decimal128 %A
  ret decimal128 %A
}
