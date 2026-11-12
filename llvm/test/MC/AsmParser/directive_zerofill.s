# RUN: llvm-mc -triple i386-apple-darwin9 %s | FileCheck %s

# CHECK: TEST0:
# CHECK: .zerofill __FOO,__bar,x,1
# CHECK: .zerofill __FOO,__bar,y,8,2
# CHECK: .zerofill __EMPTY,__NoSymbol
# CHECK: .zerofill __DATA,__bss,"what you say?",8,3
TEST0:  
	.zerofill __FOO, __bar, x, 2-1
	.zerofill __FOO,   __bar, y ,  8 , 1+1
	.zerofill __EMPTY,__NoSymbol
        
        # rdar://7965971
        .zerofill __DATA, __bss, "what you say?", 8, 3
