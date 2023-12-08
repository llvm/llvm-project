// RUN: not llvm-mc -triple x86_64-apple-darwin10 %s 2> %t.err > %t
// RUN: FileCheck --check-prefix=CHECK-OUTPUT < %t %s
// RUN: FileCheck --check-prefix=CHECK-ERROR < %t.err %s
        
// CHECK-OUTPUT: .linker_option "a"
.linker_option "a"
// CHECK-OUTPUT: .linker_option "a", "b"
.linker_option "a", "b"
// CHECK-OUTPUT-NOT: .linker_option
// CHECK-ERROR: expected string in '.linker_option' directive
// CHECK-ERROR: .linker_option 10
// CHECK-ERROR:                ^
.linker_option 10
// CHECK-ERROR: expected string in '.linker_option' directive
// CHECK-ERROR: .linker_option "a",
// CHECK-ERROR:                    ^
.linker_option "a",
// CHECK-ERROR: unexpected token in '.linker_option' directive
// CHECK-ERROR: .linker_option "a" "b"
// CHECK-ERROR:                    ^
.linker_option "a" "b"
