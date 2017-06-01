// RUN: not llvm-mc -triple armv7--none-eabi %s 2>&1 | FileCheck %s
// RUN: not llvm-mc -triple armv7--none-eabi %s 2>&1 | grep "error:" | count 33
	
// CHECK: [[@LINE+1]]:10: error: unexpected token
	.word 0 $
// CHECK-NOT: [[@LINE+1]]:{{[0-9]+}}: error:
	.word 0 # EOL COMMENT
// CHECK: [[@LINE+1]]:11: error: unexpected token
	.short 0 $
// CHECK-NOT: [[@LINE+1]]:{{[0-9]+}}: error:
	.short 0 # EOL COMMENT
// CHECK: [[@LINE+1]]:11: error: unexpected token
	.hword 0 $
// CHECK-NOT: [[@LINE+1]]:{{[0-9]+}}: error:
	.hword 0 # EOL COMMENT

  .arch armv7-a
// CHECK: [[@LINE+1]]:9: error: unexpected token in directive
	.thumb $
// CHECK-NOT: [[@LINE+1]]:{{[0-9]+}}: error:	
	.thumb # EOL COMMENT

// CHECK: [[@LINE+1]]:7: error: unexpected token in directive
	.arm $
// CHECK-NOT: [[@LINE+1]]:{{[0-9]+}}: error:	
	.arm # EOL COMMENT		
// CHECK: [[@LINE+1]]:14: error: unexpected token in '.thumb_func' directive
	.thumb_func $ 
// CHECK-NOT: [[@LINE+1]]:{{[0-9]+}}: error:	
	.thumb_func # EOL COMMENT
// CHECK: [[@LINE+1]]:11: error: unexpected token in directive	
	.code 16 $
// CHECK-NOT: [[@LINE+1]]:{{[0-9]+}}: error:
	.code 16 # EOL COMMENTS	
// CHECK: [[@LINE+1]]:18: error: unexpected token in directive	
	.syntax unified $
// CHECK-NOT: [[@LINE+1]]:{{[0-9]+}}: error:
	.syntax unified # EOL COMMENT
	fred .req r5
// CHECK: [[@LINE+1]]:14: error: unexpected input in '.unreq' directive
	.unreq fred $
// CHECK-NOT: [[@LINE+1]]:{{[0-9]+}}: error:
	.unreq fred # EOL COMMENTS
	
// CHECK: [[@LINE+1]]:18: error: unexpected token in '.fnstart' directive
        .fnstart $
// CHECK-NOT: [[@LINE+1]]:{{[0-9]+}}: error:
	.fnstart # EOL COMMENT
// CHECK: [[@LINE+1]]:23: error: unexpected token in '.cantunwind' directive
        .cantunwind   $
// CHECK-NOT: [[@LINE+1]]:{{[0-9]+}}: error:
	.cantunwind   # EOL COMMENT	


// CHECK: [[@LINE+1]]:18: error: unexpected token in '.fnend' directive
        .fnend   $
// CHECK-NOT: [[@LINE+1]]:{{[0-9]+}}: error:
	.fnend   # EOL COMMENT	

	.fnstart
// CHECK: [[@LINE+1]]:43: error: unexpected token in '.personality' directive
        .personality __gxx_personality_v0 $
// CHECK-NOT: [[@LINE+1]]:{{[0-9]+}}: error:
        .personality __gxx_personality_v0 # EOL COMMENET

// CHECK: [[@LINE+1]]:28: error: unexpected token
        .setfp  fp, sp, #0 $
// CHECK-NOT: [[@LINE+1]]:{{[0-9]+}}: error:
        .setfp  fp, sp, #0 # EOL COMMENT


// CHECK: [[@LINE+1]]:17: error: unexpected token in '.pad' directive
        .pad #0 $
// CHECK-NOT: [[@LINE+1]]:{{[0-9]+}}: error:
        .pad #0 # EOL COMMENT

// CHECK: [[@LINE+1]]:20: error: unexpected token in directive
        .save {r0} $
// CHECK-NOT: [[@LINE+1]]:{{[0-9]+}}: error:
        .save {r0} # EOL COMMENT

// CHECK: [[@LINE+1]]:21: error: unexpected token in directive
        .vsave {d0} $
// CHECK-NOT: [[@LINE+1]]:{{[0-9]+}}: error:
        .vsave {d0} # EOL COMMENT
	
		
// CHECK: [[@LINE+1]]:22: error: unexpected token in '.handlerdata' directive
        .handlerdata $
// CHECK-NOT: [[@LINE+1]]:{{[0-9]+}}: error:
        .handlerdata # EOL COMMENT

	.fnend

// CHECK: [[@LINE+1]]:9: error: unexpected token in directive
	.ltorg $
// CHECK-NOT: [[@LINE+1]]:{{[0-9]+}}: error:
	.ltorg # EOL COMMENT
// CHECK: [[@LINE+1]]:8: error: unexpected token in directive
	.pool $
// CHECK-NOT: [[@LINE+1]]:{{[0-9]+}}: error:
	.pool # EOL COMMENT
// CHECK: [[@LINE+1]]:8: error: unexpected token in directive
	.even $
// CHECK-NOT: [[@LINE+1]]:{{[0-9]+}}: error:
	.even	 # EOL COMMENT	
	.fnstart
// CHECK: [[@LINE+1]]:22: error: unexpected token in '.personalityindex' directive
	.personalityindex 0 $
// CHECK-NOT: [[@LINE+1]]:{{[0-9]+}}: error:
	.personalityindex 0 # EOL COMMENT
	.fnend

	.fnstart
// CHECK: [[@LINE+1]]:19: error: unexpected token
	.unwind_raw 0, 0 $
// CHECK-NOT: [[@LINE+1]]:{{[0-9]+}}: error:
	.unwind_raw 0, 0 # EOL COMMENT

// CHECK: [[@LINE+1]]:12: error: unexpected token in '.movsp' directive
	.movsp r0 $
// CHECK-NOT: [[@LINE+1]]:{{[0-9]+}}: error:
	.movsp r1 # EOL COMMENT
	.fnend

// CHECK: [[@LINE+1]]:21: error: unexpected token in '.arch_extension' directive
	.arch_extension mp $
// CHECK-NOT: [[@LINE+1]]:{{[0-9]+}}: error:	
	.arch_extension mp # EOL COMMENT

// CHECK: [[@LINE+1]]:21: error: unexpected token in '.arch_extension' directive
	.arch_extension mp $
// CHECK-NOT: [[@LINE+1]]:{{[0-9]+}}: error:	
	.arch_extension mp # EOL COMMENT

        .type arm_func,%function
arm_func:
        nop
// CHECK: [[@LINE+1]]:45: error: unexpected token
        .thumb_set alias_arm_func, arm_func $
// CHECK-NOT: [[@LINE+1]]:{{[0-9]+}}: error:	
        .thumb_set alias_arm_func, arm_func # EOL COMMENT

// CHECK: [[@LINE+1]]:23: error: unexpected token in '.eabi_attribute' directive
	.eabi_attribute 0, 0 $
// CHECK-NOT: [[@LINE+1]]:{{[0-9]+}}: error:		
	.eabi_attribute 0, 0 # EOL COMMENT

.arm
// CHECK: [[@LINE+1]]:10: error: unexpected token
	.inst	2 $
// CHECK-NOT: [[@LINE+1]]:{{[0-9]+}}: error:		
	.inst	2 # EOL COMMENT
.thumb
// CHECK: [[@LINE+1]]:12: error: unexpected token
	.inst.n 2 $
// CHECK-NOT: [[@LINE+1]]:{{[0-9]+}}: error:		
	.inst.n 2 # EOL COMMENT
// CHECK: [[@LINE+1]]:12: error: unexpected token
	.inst.w 4 $
// CHECK-NOT: [[@LINE+1]]:{{[0-9]+}}: error:		
	.inst.w 4 # EOL COMMENT
// CHECK: [[@LINE+1]]:21: error: unexpected token
	.object_arch armv7 $
// CHECK-NOT: [[@LINE+1]]:{{[0-9]+}}: error:		
	.object_arch armv7 # EOL COMMENT
// CHECK: [[@LINE+1]]:23: error: unexpected token in '.tlsdescseq' directive
	.tlsdescseq variable $
// CHECK-NOT: [[@LINE+1]]:{{[0-9]+}}: error:		
	.tlsdescseq variable # EOL COMMENT
