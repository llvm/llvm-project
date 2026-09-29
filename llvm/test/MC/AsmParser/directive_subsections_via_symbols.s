# RUN: llvm-mc -triple i386-apple-darwin9 %s | FileCheck %s

# CHECK: TEST0:
# CHECK: .subsections_via_symbols
TEST0:  
	.subsections_via_symbols
