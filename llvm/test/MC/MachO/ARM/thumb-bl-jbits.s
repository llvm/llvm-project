@ RUN: llvm-mc -triple=thumbv7-apple-darwin -filetype=obj -o - < %s | macho-dump --dump-section-data | FileCheck %s
.thumb
.thumb_func t
t:	nop

.data
.space 4441096 - 4 - 2

.section __TEXT, __branch, regular, pure_instructions
.thumb
.thumb_func b
b:
	bl	t
# CHECK: '_section_data', 'c3f7fcf5'
# We are checking that the branch and link instruction which is:
#	bl	#-4441096
# has it displacement encoded correctly with respect to the J1 and J2 bits when
# the branch is assembled with a label not a displacement.
# rdar://10149689
