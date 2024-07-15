// RUN: llvm-mc -filetype=obj -triple x86_64 %s 2>&1 -o /dev/null | FileCheck -DFILE=%s --strict-whitespace %s --implicit-check-not="{{[0-9]+:[0-9]+: warning: unterminated string}}" --check-prefix=CHECK-WARN

.string "abcd\xFFefg
12345678"

// CHECK-WARN:   [[#@LINE-3]]:21: warning: unterminated string; newline inserted
// CHECK-NEXT:  .string "abcd\xFFefg

.ascii "some test ascii

sequence
with
newlines\x0A
"

// CHECK-WARN:   [[#@LINE-7]]:24: warning: unterminated string; newline inserted
// CHECK-NEXT:   .ascii "some test ascii
// CHECK-WARN:   [[#@LINE-8]]:1: warning: unterminated string; newline inserted
// CHECK-WARN:   [[#@LINE-8]]:9: warning: unterminated string; newline inserted
// CHECK-NEXT:   sequence
// CHECK-WARN:   [[#@LINE-9]]:5: warning: unterminated string; newline inserted
// CHECK-NEXT:   with
// CHECK-WARN:   [[#@LINE-10]]:13: warning: unterminated string; newline inserted
// CHECK-NEXT:   newlines\x0A

.asciz "another test string

with
newline characters


"

// CHECK-WARN:   [[#@LINE-8]]:28: warning: unterminated string; newline inserted
// CHECK-NEXT:   .asciz "another test string
// CHECK-WARN:   [[#@LINE-9]]:1: warning: unterminated string; newline inserted
// CHECK-WARN:   [[#@LINE-9]]:5: warning: unterminated string; newline inserted
// CHECK-NEXT:   with
// CHECK-WARN:   [[#@LINE-10]]:19: warning: unterminated string; newline inserted
// CHECK-NEXT:   newline characters
// CHECK-WARN:   [[#@LINE-11]]:1: warning: unterminated string; newline inserted
// CHECK-WARN:   [[#@LINE-11]]:1: warning: unterminated string; newline inserted

.file "warn-newline
.s"
// CHECK-WARN:   [[#@LINE-2]]:20: warning: unterminated string; newline inserted

.cv_file 1 "some_an
other_file.s"
// CHECK-WARN:   [[#@LINE-2]]:20: warning: unterminated string; newline inserted

.ascii "test\nvalid1_string\xFF\n\n\xFF"
.asciz "\n\n\nvalid2_string\x0A"
.string "1234\nvalid3_string\xFF\n\xFF\n"
