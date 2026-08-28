; Test that -split-by-category=attribute without --category-attribute produces an error.

; RUN: not llvm-split -split-by-category=attribute -S < %s -o %t 2>&1 \
; RUN:   | FileCheck %s

; CHECK: error: -split-by-category=attribute requires --category-attribute=<name>
