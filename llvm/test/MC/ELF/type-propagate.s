// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj --symbols - | FileCheck %s

// This tests that types are propagated from symbols to their aliases. Our
// behavior is a bit different than gas. If the type of a symbol changes,
// gas will update the type of the aliases only if those aliases were declare
// at a point in the file where the aliased symbol was already define.

// The lines marked with GAS illustrate this difference.


	.type sym01, @object
sym01:
	.type sym02, @function
sym02:

	sym03 = sym01
	sym04 = sym03
.type sym03, @function
	sym05 = sym03
	sym06 = sym01 - sym02
	sym07 = sym02 - sym01

	sym08 = sym10
	sym09 = sym10 + 1
	.type sym10, @object
sym10:

	sym11 = sym10
	sym12 = sym10 + 1
	.type sym10, @function

// CHECK:       Symbol {
// CHECK:         Name: sym01
// CHECK-NEXT:    Value: 0x0
// CHECK-NEXT:    Size: 0
// CHECK-NEXT:    Binding: Local (0x0)
// CHECK-NEXT:    Type: Object (0x1)
// CHECK-NEXT:    Other: 0
// CHECK-NEXT:    Section: .text
// CHECK-NEXT:  }
// CHECK-NEXT:  Symbol {
// CHECK-NEXT:    Name: sym02
// CHECK-NEXT:    Value: 0x0
// CHECK-NEXT:    Size: 0
// CHECK-NEXT:    Binding: Local (0x0)
// CHECK-NEXT:    Type: Function (0x2)
// CHECK-NEXT:    Other: 0
// CHECK-NEXT:    Section: .text
// CHECK-NEXT:  }
// CHECK-NEXT:  Symbol {
// CHECK-NEXT:    Name: sym03
// CHECK-NEXT:    Value: 0x0
// CHECK-NEXT:    Size: 0
// CHECK-NEXT:    Binding: Local (0x0)
// CHECK-NEXT:    Type: Function (0x2)
// CHECK-NEXT:    Other: 0
// CHECK-NEXT:    Section: .text
// CHECK-NEXT:  }
// CHECK-NEXT:  Symbol {
// CHECK-NEXT:    Name: sym04
// CHECK-NEXT:    Value: 0x0
// CHECK-NEXT:    Size: 0
// CHECK-NEXT:    Binding: Local (0x0)
// CHECK-NEXT:    Type: Object (0x1)
// CHECK-NEXT:    Other: 0
// CHECK-NEXT:    Section: .text
// CHECK-NEXT:  }
// CHECK-NEXT:  Symbol {
// CHECK-NEXT:    Name: sym05
// CHECK-NEXT:    Value: 0x0
// CHECK-NEXT:    Size: 0
// CHECK-NEXT:    Binding: Local (0x0)

// GAS:           Type: Function (0x2)
// CHECK-NEXT:    Type: Object (0x1)

// CHECK-NEXT:    Other: 0
// CHECK-NEXT:    Section: .text
// CHECK-NEXT:  }
// CHECK-NEXT:  Symbol {
// CHECK-NEXT:    Name: sym06
// CHECK-NEXT:    Value: 0x0
// CHECK-NEXT:    Size: 0
// CHECK-NEXT:    Binding: Local (0x0)
// CHECK-NEXT:    Type: None (0x0)
// CHECK-NEXT:    Other: 0
// CHECK-NEXT:    Section: Absolute (0xFFF1)
// CHECK-NEXT:  }
// CHECK-NEXT:  Symbol {
// CHECK-NEXT:    Name: sym07
// CHECK-NEXT:    Value: 0x0
// CHECK-NEXT:    Size: 0
// CHECK-NEXT:    Binding: Local (0x0)
// CHECK-NEXT:    Type: None (0x0)
// CHECK-NEXT:    Other: 0
// CHECK-NEXT:    Section: Absolute (0xFFF1)
// CHECK-NEXT:  }
// CHECK-NEXT:  Symbol {
// CHECK-NEXT:    Name: sym08
// CHECK-NEXT:    Value: 0x0
// CHECK-NEXT:    Size: 0
// CHECK-NEXT:    Binding: Local (0x0)
// CHECK-NEXT:    Type: Function (0x2)
// CHECK-NEXT:    Other: 0
// CHECK-NEXT:    Section: .text
// CHECK-NEXT:  }
// CHECK-NEXT:  Symbol {
// CHECK-NEXT:    Name: sym10
// CHECK-NEXT:    Value: 0x0
// CHECK-NEXT:    Size: 0
// CHECK-NEXT:    Binding: Local (0x0)
// CHECK-NEXT:    Type: Function (0x2)
// CHECK-NEXT:    Other: 0
// CHECK-NEXT:    Section: .text
// CHECK-NEXT:  }
// CHECK-NEXT:  Symbol {
// CHECK-NEXT:    Name: sym09
// CHECK-NEXT:    Value: 0x1
// CHECK-NEXT:    Size: 0
// CHECK-NEXT:    Binding: Local (0x0)

// GAS:           Type: None (0x0)
// CHECK-NEXT:    Type: Function (0x2)

// CHECK-NEXT:    Other: 0
// CHECK-NEXT:    Section: .text
// CHECK-NEXT:  }
// CHECK-NEXT:  Symbol {
// CHECK-NEXT:    Name: sym11
// CHECK-NEXT:    Value: 0x0
// CHECK-NEXT:    Size: 0
// CHECK-NEXT:    Binding: Local (0x0)

// GAS:           Type: Object (0x1)
// CHECK-NEXT:    Type: Function (0x2)

// CHECK-NEXT:    Other: 0
// CHECK-NEXT:    Section: .text
// CHECK-NEXT:  }
// CHECK-NEXT:  Symbol {
// CHECK-NEXT:    Name: sym12
// CHECK-NEXT:    Value: 0x1
// CHECK-NEXT:    Size: 0
// CHECK-NEXT:    Binding: Local (0x0)

// GAS:           Type: Object (0x1)
// CHECK-NEXT:    Type: Function (0x2)

// CHECK-NEXT:    Other: 0
// CHECK-NEXT:    Section: .text
// CHECK-NEXT:  }
