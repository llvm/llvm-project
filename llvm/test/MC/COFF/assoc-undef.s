# RUN: not llvm-mc %s -filetype=obj -triple=x86_64-windows-msvc -o /dev/null 2>&1 | FileCheck %s

# CHECK: cannot make section assocSec associative with sectionless symbol undef

	.section	assocSec,"dr",associative,undef
	.p2align	3
	.quad	my_initializer

