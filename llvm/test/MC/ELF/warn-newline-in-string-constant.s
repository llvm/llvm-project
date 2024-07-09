// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s 2>&1 -o %t | FileCheck %s --check-prefix=CHECK-WARN

.string "abcdefg
12345678"

.ascii "some test ascii

sequence
with
newlines
"

.asciz "another test string

with
newline characters


"

// CHECK-WARN:        warn-newline-in-string-constant.s:3:17: warning: unterminated string; newline inserted
// CHECK-WARN:  .string "abcdefg

// CHECK-WARN:   warn-newline-in-string-constant.s:6:24: warning: unterminated string; newline inserted
// CHECK-WARN:  .ascii "some test ascii
// CHECK-WARN:                         ^
// CHECK-WARN:   warn-newline-in-string-constant.s:7:1: warning: unterminated string; newline inserted
// CHECK-WARN:   ^
// CHECK-WARN:   warn-newline-in-string-constant.s:8:9: warning: unterminated string; newline inserted
// CHECK-WARN:   sequence
// CHECK-WARN:           ^
// CHECK-WARN:   warn-newline-in-string-constant.s:9:5: warning: unterminated string; newline inserted
// CHECK-WARN:   with
// CHECK-WARN:        ^
// CHECK-WARN:   warn-newline-in-string-constant.s:10:9: warning: unterminated string; newline inserted
// CHECK-WARN:   newlines
// CHECK-WARN:           ^

// CHECK-WATN:   warn-newline-in-string-constant.s:13:28: warning: unterminated string; newline inserted
// CHECK-WARN:   .asciz "another test string
// CHECK-WARN:   warn-newline-in-string-constant.s:14:1: warning: unterminated string; newline inserted
// CHECK-WARN:   ^
// CHECK-WARN:   warn-newline-in-string-constant.s:15:5: warning: unterminated string; newline inserted
// CHECK-WARN:   with
// CHECK-WARN:        ^
// CHECK-WARN:   warn-newline-in-string-constant.s:16:19: warning: unterminated string; newline inserted
// CHECK-WARN:   newline characters
// CHECK-WARN:                      ^
// CHECK-WARN:   warn-newline-in-string-constant.s:17:1: warning: unterminated string; newline inserted
// CHECK-WARN:   ^
// CHECK-WARN:   warn-newline-in-string-constant.s:18:1: warning: unterminated string; newline inserted
// CHECK-WARN:   ^
