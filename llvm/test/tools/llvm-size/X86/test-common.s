// #Check that with common switch commons are added to bss or 
// #Shown as *COM* otherwise their size is discounted
// RUN: llvm-mc %s -o %t.o -filetype=obj -triple=x86_64-pc-linux
// RUN: llvm-size -A -common %t.o | FileCheck --check-prefix="SYSV" %s
// RUN: llvm-size -B -common %t.o| FileCheck --check-prefix="BSD" %s
// RUN: llvm-size -A %t.o | FileCheck --check-prefix="SYSVNOCOMM" %s
// RUN: llvm-size -B %t.o| FileCheck --check-prefix="BSDNOCOMM" %s
	.type	x,@object
	.comm	x,4,4
	.type	y,@object
	.comm	y,4,4
	.type	z,@object
	.comm	z,4,4
// SYSV:      {{[ -\(\)_A-Za-z0-9.\\/:]+}}  :
// SYSV-NEXT: section   size   addr
// SYSV-NEXT: .text          0      0
// SYSV-NEXT: *COM*     12      0
// SYSV-NEXT: Total     12

// SYSVNOCOMM:      {{[ -\(\)_A-Za-z0-9.\\/:]+}}  :
// SYSVNOCOMM-NEXT: section   size   addr
// SYSVNOCOMM-NEXT: .text          0      0
// SYSVNOCOMM-NEXT: Total      0

// BSD:      text    data     bss     dec     hex filename
// BSD-NEXT:    0       0      12       12       c  {{[ -\(\)_A-Za-z0-9.\\/:]+}}

// BSDNOCOMM:      text    data     bss     dec     hex filename
// BSDNOCOMM-NEXT:    0       0       0       0       0  {{[ -\(\)_A-Za-z0-9.\\/:]+}}
