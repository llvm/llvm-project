; RUN: llc -mtriple x86_64-unknown-linux-gnu -filetype asm -o - %s | FileCheck %s
; REQUIRES: x86-registered-target

@__Swift_AST = internal constant [8 x i8] c"test.ast", section ".swift_ast", align 4
@_swift1_autolink_entries = private constant [0 x i8] zeroinitializer, section ".swift1_autolink_entries", no_sanitize_address, align 8
@llvm.used = appending global [2 x ptr] [ptr @__Swift_AST, ptr @_swift1_autolink_entries], section "llvm.metadata"

; CHECK: .section .swift_ast,"R"
; CHECK: .section .swift1_autolink_entries,"eR"
