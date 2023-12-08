; RUN: llvm-as < %s | llvm-dis | grep "addrspace(33)" | count 7
; RUN: llvm-as < %s | llvm-dis | grep "addrspace(42)" | count 2
; RUN: llvm-as < %s | llvm-dis | grep "addrspace(66)" | count 2
; RUN: llvm-as < %s | llvm-dis | grep "addrspace(11)" | count 3
; RUN: llvm-as < %s | llvm-dis | grep "addrspace(22)" | count 4
; RUN: verify-uselistorder %s

	%struct.mystruct = type { i32, ptr addrspace(33), i32, ptr addrspace(33) }
@input = weak addrspace(42) global %struct.mystruct zeroinitializer  		; <ptr addrspace(42)> [#uses=1]
@output = addrspace(66) global %struct.mystruct zeroinitializer 		; <ptr addrspace(66)> [#uses=1]
@y = external addrspace(33) global ptr addrspace(22) 		; <ptr addrspace(33)> [#uses=1]

define void @foo() {
entry:
	%tmp1 = load ptr addrspace(33), ptr addrspace(42) getelementptr (%struct.mystruct, ptr addrspace(42) @input, i32 0, i32 3), align 4		; <ptr addrspace(33)> [#uses=1]
	store ptr addrspace(33) %tmp1, ptr addrspace(66) getelementptr (%struct.mystruct, ptr addrspace(66) @output, i32 0, i32 1), align 4
	ret void
}

define ptr addrspace(11) @bar(ptr addrspace(33) %x) {
entry:
	%tmp1 = load ptr addrspace(22), ptr addrspace(33) @y, align 4		; <ptr addrspace(22)> [#uses=2]
	store ptr addrspace(22) %tmp1, ptr addrspace(33) %x, align 4
	%tmp5 = load ptr addrspace(11), ptr addrspace(22) %tmp1, align 4		; <ptr addrspace(11)> [#uses=1]
	ret ptr addrspace(11) %tmp5
}
