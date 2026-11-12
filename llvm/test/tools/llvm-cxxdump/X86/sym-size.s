// RUN: llvm-mc %s -o %t -filetype=obj -triple=x86_64-pc-win32
// RUN: llvm-cxxdump %t | FileCheck %s

// CHECK:      ??_8B@@7B@[0]: 8
// CHECK-NEXT: ??_8B@@7B@[4]: 9
// CHECK-NEXT: ??_8C@@7B@[0]: 10
// CHECK-NEXT: ??_8C@@7B@[4]: 11
// CHECK-NEXT: ??_8D@@7B0@@[0]: 0
// CHECK-NEXT: ??_8D@@7B0@@[4]: 1
// CHECK-NEXT: ??_8D@@7B0@@[8]: 2
// CHECK-NEXT: ??_8D@@7B0@@[12]: 3
// CHECK-NEXT: ??_8D@@7BB@@@[0]: 4
// CHECK-NEXT: ??_8D@@7BB@@@[4]: 5
// CHECK-NEXT: ??_8D@@7BC@@@[0]: 6
// CHECK-NEXT: ??_8D@@7BC@@@[4]: 7
// CHECK-NEXT: ??_8XYZ[0]: 10
// CHECK-NEXT: ??_8XYZ[4]: 11

	.section	.rdata,"dr"
	.globl	"??_8D@@7B0@@"
"??_8D@@7B0@@":
	.long	0
	.long	1
	.long	2
	.long	3

	.globl	"??_8D@@7BB@@@"
"??_8D@@7BB@@@":
	.long	4
	.long	5

	.globl	"??_8D@@7BC@@@"
"??_8D@@7BC@@@":
	.long	6
	.long	7

	.globl	"??_8B@@7B@"
"??_8B@@7B@":
	.long	8
	.long	9

	.globl	"??_8C@@7B@"
"??_8C@@7B@":
	.long	10
	.long	11

"??_8XYZ" = "??_8C@@7B@"
