// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s 2>&1 -o /dev/null | FileCheck --strict-whitespace %s --check-prefix=CHECK-WARN

.string "abcd\xFFefg
12345678"

// CHECK-WARN:   warn-newline-in-escaped-string.s:[[#@LINE-3]]:21: warning: unterminated string; newline inserted
// CHECK-WARN:  .string "abcd\xFFefg

.ascii "some test ascii

sequence
with
newlines\x0A
"

// CHECK-WARN:   warn-newline-in-escaped-string.s:[[#@LINE-7]]:24: warning: unterminated string; newline inserted
// CHECK-WARN:  .ascii "some test ascii
// CHECK-WARN:                         ^
// CHECK-WARN:   warn-newline-in-escaped-string.s:[[#@LINE-9]]:1: warning: unterminated string; newline inserted
// CHECK-WARN:   ^
// CHECK-WARN:   warn-newline-in-escaped-string.s:[[#@LINE-10]]:9: warning: unterminated string; newline inserted
// CHECK-WARN:   sequence
// CHECK-WARN:           ^
// CHECK-WARN:   warn-newline-in-escaped-string.s:[[#@LINE-12]]:5: warning: unterminated string; newline inserted
// CHECK-WARN:   with
// CHECK-WARN:        ^
// CHECK-WARN:   warn-newline-in-escaped-string.s:[[#@LINE-14]]:13: warning: unterminated string; newline inserted
// CHECK-WARN:   newlines\x0A
// CHECK-WARN:           ^

.asciz "another test string

with
newline characters


"

// CHECK-WARN:   warn-newline-in-escaped-string.s:[[#@LINE-8]]:28: warning: unterminated string; newline inserted
// CHECK-WARN:   .asciz "another test string
// CHECK-WARN:   warn-newline-in-escaped-string.s:[[#@LINE-9]]:1: warning: unterminated string; newline inserted
// CHECK-WARN:   ^
// CHECK-WARN:   warn-newline-in-escaped-string.s:[[#@LINE-10]]:5: warning: unterminated string; newline inserted
// CHECK-WARN:   with
// CHECK-WARN:        ^
// CHECK-WARN:   warn-newline-in-escaped-string.s:[[#@LINE-12]]:19: warning: unterminated string; newline inserted
// CHECK-WARN:   newline characters
// CHECK-WARN:                      ^
// CHECK-WARN:   warn-newline-in-escaped-string.s:[[#@LINE-14]]:1: warning: unterminated string; newline inserted
// CHECK-WARN:   ^
// CHECK-WARN:   warn-newline-in-escaped-string.s:[[#@LINE-15]]:1: warning: unterminated string; newline inserted
// CHECK-WARN:   ^

.file "warn-newline
.s"
// CHECK-WARN:   warn-newline-in-escaped-string.s:[[#@LINE-2]]:20: warning: unterminated string; newline inserted

.cv_file 1 "some_an
other_file.s"
// CHECK-WARN:   warn-newline-in-escaped-string.s:[[#@LINE-2]]:20: warning: unterminated string; newline inserted

.ascii "test\nstring\xFF\n\n\xFF"
// CHECK-WARN-NOT:    warn-newline-in-escaped-string.s:[[#@LINE-1]]{{.*}}

.asciz "\n\n\ntest_string\x0A"
// CHECK-WARN-NOT:    warn-newline-in-escaped-string.s:[[#@LINE-1]]{{.*}}

.string "1234\n\xFF\n\xFF\n"
// CHECK-WARN-NOT:    warn-newline-in-escaped-string.s:[[#@LINE-1]]{{.*}}
