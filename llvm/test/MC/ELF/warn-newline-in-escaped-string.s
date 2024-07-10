// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s 2>&1 -o /dev/null | FileCheck -DFILE=%s --strict-whitespace %s --implicit-check-not=valid1_string --implicit-check-not=valid2_string --implicit-check-not=valid3_string --check-prefix=CHECK-WARN

.string "abcd\xFFefg
12345678"

// CHECK-WARN:   warn-newline-in-escaped-string.s:[[#@LINE-3]]:21: warning: unterminated string; newline inserted
// CHECK-NEXT:  .string "abcd\xFFefg

.ascii "some test ascii

sequence
with
newlines\x0A
"

// CHECK-NEXT:   [[#@LINE-7]]:24: warning: unterminated string; newline inserted
// CHECK-NEXT:  .ascii "some test ascii
// CHECK-NEXT:   [[#@LINE-7]]:9: warning: unterminated string; newline inserted
// CHECK-NEXT:   sequence
// CHECK-NEXT:   [[#@LINE-8]]:5: warning: unterminated string; newline inserted
// CHECK-NEXT:   with
// CHECK-NEXT:   [[#@LINE-9]]:13: warning: unterminated string; newline inserted
// CHECK-NEXT:   newlines\x0A

.asciz "another test string

with
newline characters


"

// CHECK-NEXT:   [[#@LINE-8]]:28: warning: unterminated string; newline inserted
// CHECK-NEXT:   .asciz "another test string
// CHECK-NEXT:   [[#@LINE-8]]:5: warning: unterminated string; newline inserted
// CHECK-NEXT:   with
// CHECK-NEXT:   [[#@LINE-9]]:19: warning: unterminated string; newline inserted
// CHECK-NEXT:   newline characters

.file "warn-newline
.s"
// CHECK-NEXT:   [[#@LINE-2]]:20: warning: unterminated string; newline inserted

.cv_file 1 "some_an
other_file.s"
// CHECK-NEXT:   [[#@LINE-2]]:20: warning: unterminated string; newline inserted

.ascii "test\nvalid1_string\xFF\n\n\xFF"
.asciz "\n\n\nvalid2_string\x0A"
.string "1234\nvalid3_string\xFF\n\xFF\n"
