// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s 2>&1 -o %t | FileCheck --strict-whitespace %s --check-prefix=CHECK-WARN

.string "abcd\xFFefg
12345678"

.ascii "some test ascii

sequence
with
newlines\x0A
"

.asciz "another test string

with
newline characters


"

// CHECK-WARN:        warn-newline-in-string-constant.s:3:21: warning: unterminated string; newline inserted
// CHECK-WARN:  .string "abcd\xFFefg

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
// CHECK-WARN:   warn-newline-in-string-constant.s:10:13: warning: unterminated string; newline inserted
// CHECK-WARN:   newlines\x0A
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

.ascii "test\nstring\xFF\n\n\xFF"

// CHECK-WARN-NOT: warn-newline-in-string-constant.s:55{{.*}}

.asciz "\n\n\ntest_string\x0A"

// CHECK-WARN-NOT: warn-newline-in-string-constant.s:59{{.*}}

.string "1234\n\xFF\n\xFF\n"

// CHECK-WARN-NOT: warn-newline-in-string-constant.s:63{{.*}}