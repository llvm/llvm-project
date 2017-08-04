; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s

@g1 = global i32 7 "key" = "value" "key2" = "value2"
@g2 = global i32 2, align 4 "key3" = "value3"
@g3 = global i32 2 #0
@g4 = global i32 2, align 4 "key5" = "value5" #0

attributes #0 = { "string" = "value" nobuiltin norecurse }

; CHECK: @g1 = global i32 7 #0
; CHECK: @g2 = global i32 2, align 4 #1
; CHECK: @g3 = global i32 2 #2
; CHECK: @g4 = global i32 2, align 4 #3

; CHECK: attributes #0 = { "key"="value" "key2"="value2" }
; CHECK: attributes #1 = { "key3"="value3" }
; CHECK: attributes #2 = { nobuiltin norecurse "string"="value" }
; CHECK: attributes #3 = { nobuiltin norecurse "key5"="value5" "string"="value" }

