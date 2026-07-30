; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s

@g1 = global i32 7 "key" = "value" "key2" = "value2"
@g2 = global i32 2, align 4 "key3" = "value3"
@g3 = global i32 2 #0
@g4 = global i32 2, align 4 "key5" = "value5" #0
@g5 = global i32 2, no_sanitize_address, align 4
@g6 = global i32 2, no_sanitize_hwaddress, align 4
@g7 = global i32 2, sanitize_address_dyninit, align 4
@g8 = global i32 2, sanitize_memtag, align 4
@g9 = global i32 2, no_sanitize_address, no_sanitize_hwaddress, sanitize_memtag, align 4
@g10 = global i32 2, code_model "tiny"
@g11 = global i32 2, code_model "small"
@g12 = global i32 2, code_model "kernel"
@g13 = global i32 2, code_model "medium"
@g14 = global i32 2, code_model "large"

attributes #0 = { "string" = "value" nobuiltin norecurse }

; CHECK: @g1 = global i32 7 #0
; CHECK: @g2 = global i32 2, align 4 #1
; CHECK: @g3 = global i32 2 #2
; CHECK: @g4 = global i32 2, align 4 #3
; CHECK: @g5 = global i32 2, no_sanitize_address, align 4
; CHECK: @g6 = global i32 2, no_sanitize_hwaddress, align 4
; CHECK: @g7 = global i32 2, sanitize_address_dyninit, align 4
; CHECK: @g8 = global i32 2, sanitize_memtag, align 4
; CHECK: @g9 = global i32 2, no_sanitize_address, no_sanitize_hwaddress, sanitize_memtag, align 4
; CHECK: @g10 = global i32 2, code_model "tiny"
; CHECK: @g11 = global i32 2, code_model "small"
; CHECK: @g12 = global i32 2, code_model "kernel"
; CHECK: @g13 = global i32 2, code_model "medium"
; CHECK: @g14 = global i32 2, code_model "large"

; CHECK: attributes #0 = { "key"="value" "key2"="value2" }
; CHECK: attributes #1 = { "key3"="value3" }
; CHECK: attributes #2 = { nobuiltin norecurse "string"="value" }
; CHECK: attributes #3 = { nobuiltin norecurse "key5"="value5" "string"="value" }

