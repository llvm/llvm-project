// RUN: llvm-mc -triple i386-unknown-unknown-code16 -show-encoding %s | FileCheck --check-prefix=CHECK16 %s
// RUN: llvm-mc -triple i386-unknown-unknown -show-encoding %s | FileCheck --check-prefix=CHECK %s
// RUN: llvm-mc -triple i686-unknown-unknown -show-encoding %s | FileCheck --check-prefix=CHECK %s	
	
.intel_syntax
	
push 0
push -1
push 30
push 257
push 65536

//CHECK16:	pushw	$0                      # encoding: [0x6a,0x00]
//CHECK16:	pushw	$-1	                # encoding: [0x6a,0xff]
//CHECK16:	pushw	$30                     # encoding: [0x6a,0x1e]
//CHECK16:	pushw	$257                    # encoding: [0x68,0x01,0x01]
//CHECK16:	pushl	$65536                  # encoding: [0x66,0x68,0x00,0x00,0x01,0x00]

//CHECK:	pushl	$0                      # encoding: [0x6a,0x00]
//CHECK:	pushl	$-1                     # encoding: [0x6a,0xff]
//CHECK:	pushl	$30                     # encoding: [0x6a,0x1e]
//CHECK:	pushl	$257                    # encoding: [0x68,0x01,0x01,0x00,0x00]
//CHECK:	pushl	$65536                  # encoding: [0x68,0x00,0x00,0x01,0x00]
