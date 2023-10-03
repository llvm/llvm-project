# RUN: not llvm-mc -triple s390x-unknown-unknown -mcpu=z13 -filetype=obj %s 2>&1 | FileCheck %s

	.text

# CHECK:      error: Unsupported absolute address
# CHECK-NEXT:        vleg %v0,0,src
# CHECK-NEXT:        ^
	vleg %v0,0,src

# CHECK:      error: Unsupported absolute address
# CHECK-NEXT:        vleih %v0,0,src
# CHECK-NEXT:        ^
	vleih %v0,0,src

# CHECK:      error: Unsupported absolute address
# CHECK-NEXT:        vleif %v0,0,src
# CHECK-NEXT:        ^
	vleif %v0,0,src

# CHECK:      error: Unsupported absolute address
# CHECK-NEXT:        vrepi %v0,0,src
# CHECK-NEXT:        ^
	vrepi %v0,0,src
