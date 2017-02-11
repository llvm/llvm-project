# RUN: llvm-mc -triple=hexagon -filetype=obj %s | llvm-objdump -d -r - | FileCheck %s
#

# Verify that capitaizled endloops work

	{ R0 = mpyi(R0,R0) } : endloop0
	{ R0 = mpyi(R0,R0) } : ENDLOOP0
	{ R0 = mpyi(R0,R0) }:endloop0

	{ R0 = mpyi(R0,R0) } : endloop1
	{ R0 = mpyi(R0,R0) } : ENDLOOP1
	{ R0 = mpyi(R0,R0) }:endloop1

	{ R0 = mpyi(R0,R0) } : endloop0 : endloop1
	{ R0 = mpyi(R0,R0) } : ENDLOOP0 : ENDLOOP1
	{ R0 = mpyi(R0,R0) }:endloop0:endloop1

# CHECK: r0 = mpyi(r0,r0)
# CHECK: :endloop0
# CHECK: :endloop0
# CHECK: :endloop0
# CHECK: :endloop1
# CHECK: :endloop1
# CHECK: :endloop1
# CHECK: :endloop0 :endloop1
# CHECK: :endloop0 :endloop1
# CHECK: :endloop0 :endloop1


