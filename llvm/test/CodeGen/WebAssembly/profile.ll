; RUN: llc < %s --filetype=obj | obj2yaml | FileCheck %s

target triple = "wasm32-unknown-unknown"

$__covrec_A = comdat any
$__covrec_B = comdat any

@__covrec_A = linkonce_odr hidden constant <{ i64, i32, i64, i64, [4 x i8] }> <{
    i64 -1978722966671112904,
    i32 4,
    i64 0,
    i64 -8102528905418564625,
    [4 x i8] c"\01\01\04\11"
}>, section "__llvm_covfun", comdat, align 8
@__covrec_B = linkonce_odr hidden constant <{ i64, i32, i64, i64, [4 x i8] }> <{
    i64 8006510647218728891,
    i32 9,
    i64 0,
    i64 -8102528905418564625,
    [4 x i8] c"\01\01\00\01"
}>, section "__llvm_covfun", comdat, align 8
@__llvm_coverage_mapping = private constant { { i32, i32, i32, i32 }, [4 x i8] } {
    { i32, i32, i32, i32 } { i32 0, i32 87, i32 0, i32 5 },
    [4 x i8] c"\01\01\00\02"
}, section "__llvm_covmap", align 8

; CHECK:      - Type:            CUSTOM
; CHECK-NEXT:   Name:            __llvm_covfun
; CHECK-NEXT:   Payload:         3845A90EF2298AE4040000000000000000000000EF1B31BAE3088E8F01010411
; CHECK-NEXT: - Type:            CUSTOM
; CHECK-NEXT:   Name:            __llvm_covfun
; CHECK-NEXT:   Payload:         BBEFDA6903D71C6F090000000000000000000000EF1B31BAE3088E8F01010001
; CHECK-NEXT: - Type:            CUSTOM
; CHECK-NEXT:   Name:            __llvm_covmap
; CHECK-NEXT:   Payload:         '0000000057000000000000000500000001010002'
; CHECK-NEXT: - Type:            CUSTOM
; CHECK-NEXT:   Name:            linking
; CHECK-NEXT:   Version:         2
; CHECK-NEXT:   Comdats:
; CHECK-NEXT:     - Name:            __covrec_A
; CHECK-NEXT:       Entries:
; CHECK-NEXT:         - Kind:            SECTION
; CHECK-NEXT:           Index:           1
; CHECK-NEXT:     - Name:            __covrec_B
; CHECK-NEXT:       Entries:
; CHECK-NEXT:         - Kind:            SECTION
; CHECK-NEXT:           Index:           2
