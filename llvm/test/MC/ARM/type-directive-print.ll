; RUN: llc < %s -mtriple=armv7a | FileCheck %s

module asm ".type test_notype, %notype"
module asm ".type test_function, %function"
module asm ".type test_object, %object"
module asm ".type test_common, %common"
module asm ".type test_tls_object, %tls_object"
module asm ".type test_gnu_indirect_function, %gnu_indirect_function"
module asm ".type test_gnu_unique_object, %gnu_unique_object"

; CHECK: .type test_notype,%notype
; CHECK: .type test_function,%function
; CHECK: .type test_object,%object
; CHECK: .type test_common,%common
; CHECK: .type test_tls_object,%tls_object
; CHECK: .type test_gnu_indirect_function,%gnu_indirect_function
; CHECK: .type test_gnu_unique_object,%gnu_unique_object
