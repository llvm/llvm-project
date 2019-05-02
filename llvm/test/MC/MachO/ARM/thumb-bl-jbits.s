@ RUN: llvm-mc -triple=thumbv7-apple-darwin -filetype=obj -o - < %s | llvm-readobj -S --sd | FileCheck %s
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
# We are checking that the branch and link instruction which is:
#	bl	#-4441096
# has it displacement encoded correctly with respect to the J1 and J2 bits when
# the branch is assembled with a label not a displacement.
# rdar://10149689

# CHECK: Section {
# CHECK:   Index: 2
# CHECK:   Name: __branch (5F 5F 62 72 61 6E 63 68 00 00 00 00 00 00 00 00)
# CHECK:   Segment: __TEXT (5F 5F 54 45 58 54 00 00 00 00 00 00 00 00 00 00)
# CHECK:   SectionData (
# CHECK:     0000: C3F7FCF5                             |....|
# CHECK:   )
# CHECK: }
