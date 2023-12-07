// RUN: not llvm-mc -triple aarch64 %s 2>&1 | FileCheck %s
// RUN: not llvm-mc -triple aarch64 %s 2>&1 | grep "error:" | count 60

	// CHECK: [[#@LINE+1]]:19: error: expected newline
	.equ   ident1, 0 $
	// CHECK-NOT: :[[#@LINE+1]]:{{[0-9]+}}: error:
	.equ   ident1, 0 // EOL COMMENT
	// CHECK: [[#@LINE+1]]:19: error: expected newline
	.equiv ident2, 0 $
	// CHECK-NOT: :[[#@LINE+1]]:{{[0-9]+}}: error:
	.equiv ident2, 0 // EOL COMMENT
	// CHECK: [[#@LINE+1]]:19: error: expected newline
	.set   ident3, 0 $
	// CHECK-NOT: :[[#@LINE+1]]:{{[0-9]+}}: error:
	.set   ident3, 0 // EOL COMMENT
	// CHECK: :[[#@LINE+1]]:20: error: unexpected token
	.ascii  "string1" $
	// CHECK-NOT: :[[#@LINE+1]]:{{[0-9]+}}: error:
	.ascii  "string1" // EOL COMMENT
	// CHECK: :[[#@LINE+1]]:20: error: unexpected token
	.asciz  "string2" $
	// CHECK-NOT: :[[#@LINE+1]]:{{[0-9]+}}: error:
	.asciz  "string2" // EOL COMMENT
	// CHECK: :[[#@LINE+1]]:20: error: unexpected token
	.string "string3" $	
	// CHECK-NOT: :[[#@LINE+1]]:{{[0-9]+}}: error:
	.string "string3" // EOL COMMENT	
	// CHECK: :[[#@LINE+1]]:10: error: unexpected token
	.byte 0 $
	// CHECK-NOT: :[[#@LINE+1]]:{{[0-9]+}}: error:
	.byte 0 // EOL COMMENT
	// CHECK: :[[#@LINE+1]]:10: error: unexpected token
	.dc.b 0 $
	// CHECK-NOT: :[[#@LINE+1]]:{{[0-9]+}}: error:
	.dc.b 0 // EOL COMMENT
	// CHECK: :[[#@LINE+1]]:8: error: unexpected token
	.dc 0 $
	// CHECK-NOT: :[[#@LINE+1]]:{{[0-9]+}}: error:
	.dc.b 0 // EOL COMMENT
	// CHECK: :[[#@LINE+1]]:10: error: unexpected token
	.dc.w 0 $
	// CHECK-NOT: :[[#@LINE+1]]:{{[0-9]+}}: error:
	.dc.w 0 // EOL COMMENT
	// CHECK: :[[#@LINE+1]]:11: error: unexpected token
	.short 0 $
	// CHECK-NOT: :[[#@LINE+1]]:{{[0-9]+}}: error:
	.short 0 // EOL COMMENT
	// CHECK: :[[#@LINE+1]]:11: error: unexpected token
	.value 0 $
	// CHECK-NOT: :[[#@LINE+1]]:{{[0-9]+}}: error:
	.value 0 // EOL COMMENT
	// CHECK: :[[#@LINE+1]]:11: error: unexpected token
	.2byte 0 $
	// CHECK-NOT: :[[#@LINE+1]]:{{[0-9]+}}: error:
	.2byte 0 // EOL COMMENT
	// CHECK: :[[#@LINE+1]]:10: error: unexpected token
	.long 0 $
	// CHECK-NOT: :[[#@LINE+1]]:{{[0-9]+}}: error:
	.long 0 // EOL COMMENT
	// CHECK: :[[#@LINE+1]]:10: error: unexpected token
	.int  0 $
	// CHECK-NOT: :[[#@LINE+1]]:{{[0-9]+}}: error:
	.int  0 // EOL COMMENT
	// CHECK: :[[#@LINE+1]]:11: error: unexpected token
	.4byte 0 $
	// CHECK-NOT: :[[#@LINE+1]]:{{[0-9]+}}: error:
	.4byte 0 // EOL COMMENT
	// CHECK: :[[#@LINE+1]]:10: error: unexpected token
	.dc.l 0 $
	// CHECK-NOT: :[[#@LINE+1]]:{{[0-9]+}}: error:
	.dc.l 0 // EOL COMMENT
	// CHECK: :[[#@LINE+1]]:10: error: unexpected token
	.quad 0 $
	// CHECK-NOT: :[[#@LINE+1]]:{{[0-9]+}}: error:
	.quad 0 // EOL COMMENT
	// CHECK: :[[#@LINE+1]]:11: error: unexpected token
	.8byte 0 $
	// CHECK-NOT: :[[#@LINE+1]]:{{[0-9]+}}: error:
	.8byte 0 // EOL COMMENT
	// CHECK: :[[#@LINE+1]]:10: error: unexpected token
	.dc.a 0 $
	// CHECK-NOT: :[[#@LINE+1]]:{{[0-9]+}}: error:
	.dc.a 0 // EOL COMMENT
	// CHECK: :[[#@LINE+1]]:10: error: unexpected token
	.octa 0 $
	// CHECK-NOT: :[[#@LINE+1]]:{{[0-9]+}}: error:
	.octa 0 // EOL COMMENT
	// CHECK: :[[#@LINE+1]]:12: error: unexpected token
	.single 0 $
	// CHECK-NOT: :[[#@LINE+1]]:{{[0-9]+}}: error:
	.single 0 // EOL COMMENT
	// CHECK: :[[#@LINE+1]]:11: error: unexpected token
	.float 0 $
	// CHECK-NOT: :[[#@LINE+1]]:{{[0-9]+}}: error:
	.float 0 // EOL COMMENT
	// CHECK: :[[#@LINE+1]]:10: error: unexpected token
	.dc.s 0 $
	// CHECK-NOT: :[[#@LINE+1]]:{{[0-9]+}}: error:
	.dc.s 0 // EOL COMMENT
	// CHECK: :[[#@LINE+1]]:12: error: unexpected token
	.double 0 $	
	// CHECK-NOT: :[[#@LINE+1]]:{{[0-9]+}}: error:
	.double 0 // EOL COMMENT	
	// CHECK: :[[#@LINE+1]]:10: error: unexpected token
	.dc.d 0 $		
	// CHECK-NOT: :[[#@LINE+1]]:{{[0-9]+}}: error:
	.dc.d 0 // EOL COMMENT		
	// CHECK: [[#@LINE+1]]:13: error: expected newline
	.fill 1, 1 $
	// CHECK-NOT: :[[#@LINE+1]]:{{[0-9]+}}: error:
	.fill 1, 1 // EOL COMMENT
	// CHECK: [[#@LINE+1]]:17: error: expected newline
	.fill 1, 1, 10 $
	// CHECK-NOT: :[[#@LINE+1]]:{{[0-9]+}}: error:
	.fill 1, 1, 10 // EOL COMMENT
	// CHECK: [[#@LINE+1]]:16: error: expected newline
        .org 1 $
	// CHECK-NOT: :[[#@LINE+1]]:{{[0-9]+}}: error:
        .org 1 // EOL COMMENT
	// CHECK: [[#@LINE+1]]:11: error: expected newline
	.align 0 $
	// CHECK-NOT: :[[#@LINE+1]]:{{[0-9]+}}: error:
	.align 0 // EOL COMMENT
	// CHECK: [[#@LINE+1]]:13: error: expected newline
	.align32 0 $
	// CHECK-NOT: :[[#@LINE+1]]:{{[0-9]+}}: error:
	.align32 0 // EOL COMMENT
	// CHECK: [[#@LINE+1]]:12: error: expected newline
	.balign 0 $
	// CHECK-NOT: :[[#@LINE+1]]:{{[0-9]+}}: error:
	.balign 0 // EOL COMMENT
	// CHECK: [[#@LINE+1]]:13: error: expected newline
	.balignw 0 $
	// CHECK-NOT: :[[#@LINE+1]]:{{[0-9]+}}: error:
	.balignw 0 // EOL COMMENT
	// CHECK: [[#@LINE+1]]:13: error: expected newline
	.balignl 0 $
	// CHECK-NOT: :[[#@LINE+1]]:{{[0-9]+}}: error:
	.balignl 0 // EOL COMMENT
	// CHECK: [[#@LINE+1]]:13: error: expected newline
	.p2align 0 $
	// CHECK-NOT: :[[#@LINE+1]]:{{[0-9]+}}: error:
	.p2align 0 // EOL COMMENT
	// CHECK: [[#@LINE+1]]:14: error: expected newline
	.p2alignw 0 $
	// CHECK-NOT: :[[#@LINE+1]]:{{[0-9]+}}: error:
	.p2alignw 0 // EOL COMMENT
	// CHECK: [[#@LINE+1]]:14: error: expected newline
	.p2alignl 0 $
	// CHECK-NOT: :[[#@LINE+1]]:{{[0-9]+}}: error:
	.p2alignl 0 // EOL COMMENT
	// CHECK: [[#@LINE+1]]:8: error: expected newline
	.line $
	// CHECK-NOT: :[[#@LINE+1]]:{{[0-9]+}}: error:
	.line // EOL COMMENT
	// CHECK: [[#@LINE+1]]:10: error: expected newline
	.line 0 $
	// CHECK-NOT: [[#@LINE+1]]:{{[0-9]+}}: error:
	.line 0 // EOL COMMENT

	.file 1 "hello"
	// CHECK: :[[#@LINE+1]]:16: error: unexpected token
        .loc 1 $
	// CHECK-NOT: :[[#@LINE+1]]:{{[0-9]+}}: error:
        .loc 1 // EOL COMMENT	

	// CHECK: :[[#@LINE+1]]:21: error: unexpected token
	.cv_file 1 "hello" $
	// CHECK-NOT: :[[#@LINE+1]]:{{[0-9]+}}: error:
	.cv_file 1 "hello" // EOL COMMENT

	.cv_func_id 1
	// CHECK: :[[#@LINE+1]]:14: error: unexpected token
	.cv_loc 1 1 $
	// CHECK-NOT: :[[#@LINE+1]]:{{[0-9]+}}: error:
	.cv_loc 1 1 // EOL COMMENT
	
	// CHECK: [[#@LINE+1]]:28: error: expected newline
	.bundle_lock align_to_end $
	// CHECK-NOT: [[#@LINE+1]]:{{[0-9]+}}: error:
	.bundle_lock align_to_end // EOL COMMENT	
	
	// CHECK: [[#@LINE+1]]:11: error: invalid token in expression
	.sleb128 $
	// CHECK-NOT: :[[#@LINE+1]]:{{[0-9]+}}: error:
	.sleb128 // EOL COMMENT
	// CHECK: :[[#@LINE+1]]:13: error: unexpected token
	.sleb128 0 $
	// CHECK-NOT: :[[#@LINE+1]]:{{[0-9]+}}: error:
	.sleb128 0 // EOL COMMENT

	// CHECK: :[[#@LINE+1]]:11: error: invalid token in expression
	.uleb128 $	
	// CHECK-NOT: :[[#@LINE+1]]:{{[0-9]+}}: error:
	.uleb128 // EOL COMMENT
	// CHECK: :[[#@LINE+1]]:13: error: unexpected token
	.uleb128 0 $
	// CHECK-NOT: :[[#@LINE+1]]:{{[0-9]+}}: error:
	.uleb128 0 // EOL COMMENT
	// CHECK: :[[#@LINE+1]]:31: error: unexpected token
	.globl a1                    $
	// CHECK-NOT: :[[#@LINE+1]]:{{[0-9]+}}: error:
	.globl a1                    // EOL COMMENT
	// CHECK: :[[#@LINE+1]]:31: error: unexpected token
	.global a2                   $
	// CHECK-NOT: :[[#@LINE+1]]:{{[0-9]+}}: error:
	.global a2                   // EOL COMMENT
	// CHECK: :[[#@LINE+1]]:31: error: unexpected token
	.lazy_reference a3           $
	// CHECK-NOT: :[[#@LINE+1]]:{{[0-9]+}}: error:
	.lazy_reference a3           // EOL COMMENT
	// CHECK: :[[#@LINE+1]]:31: error: unexpected token
	.symbol_resolver a4          $
	// CHECK-NOT: :[[#@LINE+1]]:{{[0-9]+}}: error:
	.symbol_resolver a4          // EOL COMMENT
	// CHECK: :[[#@LINE+1]]:31: error: unexpected token
	.private_extern a5           $
	// CHECK-NOT: :[[#@LINE+1]]:{{[0-9]+}}: error:
	.private_extern a5           // EOL COMMENT
	// CHECK: :[[#@LINE+1]]:31: error: unexpected token
	.reference a6                $
	// CHECK-NOT: :[[#@LINE+1]]:{{[0-9]+}}: error:
	.reference a6                // EOL COMMENT
	// CHECK: :[[#@LINE+1]]:31: error: unexpected token
	.weak_definition a7          $
	// CHECK-NOT: :[[#@LINE+1]]:{{[0-9]+}}: error:
	.weak_definition a7          // EOL COMMENT
	// CHECK: :[[#@LINE+1]]:31: error: unexpected token
	.weak_reference a8           $
	// CHECK-NOT: :[[#@LINE+1]]:{{[0-9]+}}: error:
	.weak_reference a8           // EOL COMMENT
	// CHECK: :[[#@LINE+1]]:31: error: unexpected token
	.weak_def_can_be_hidden a9   $
	// CHECK-NOT: :[[#@LINE+1]]:{{[0-9]+}}: error:
	.weak_def_can_be_hidden a9   // EOL COMMENT	
	// CHECK: :[[#@LINE+1]]:12: error: .warning argument must be a string
	.warning  $
	// CHECK-NOT: :[[#@LINE+1]]:{{[0-9]+}}: error:
	.warning  // EOL COMMENT
	// CHECK: [[#@LINE+1]]:21: error: expected newline
	.warning "warning" $
	// CHECK-NOT: [[#@LINE+1]]:{{[0-9]+}}: error:
	.warning "warning" // EOL COMMENT


	// CHECK: :[[#@LINE+1]]:17: error: unexpected token
	.cfi_startproc $
	// CHECK-NOT: :[[#@LINE+1]]:{{[0-9]+}}: error:
	.cfi_startproc // EOL COMMENT
	.cfi_endproc
	// CHECK: [[#@LINE+1]]:24: error: expected newline
	.cfi_startproc simple $
	// CHECK-NOT: :[[#@LINE+1]]:{{[0-9]+}}: error:
	.cfi_startproc simple // EOL COMMENT
	.cfi_endproc

	
	// CHECK-NOT: error:	 
