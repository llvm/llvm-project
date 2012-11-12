; RUN: llvm-as < %s | llvm-dis | grep "addrspace(33)" | count 7
; RUN: llvm-as < %s | llvm-dis | grep "addrspace(42)" | count 2
; RUN: llvm-as < %s | llvm-dis | grep "addrspace(66)" | count 2
; RUN: llvm-as < %s | llvm-dis | grep "addrspace(11)" | count 6
; RUN: llvm-as < %s | llvm-dis | grep "addrspace(22)" | count 5

	%struct.mystruct = type { i32, i32 addrspace(33)*, i32, i32 addrspace(33)* }
@input = weak addrspace(42) global %struct.mystruct zeroinitializer  		; <%struct.mystruct addrspace(42)*> [#uses=1]
@output = addrspace(66) global %struct.mystruct zeroinitializer 		; <%struct.mystruct addrspace(66)*> [#uses=1]
@y = external addrspace(33) global i32 addrspace(11)* addrspace(22)* 		; <i32 addrspace(11)* addrspace(22)* addrspace(33)*> [#uses=1]

define void @foo() {
entry:
	%tmp1 = load i32 addrspace(33)* addrspace(42)* getelementptr (%struct.mystruct addrspace(42)* @input, i32 0, i32 3), align 4		; <i32 addrspace(33)*> [#uses=1]
	store i32 addrspace(33)* %tmp1, i32 addrspace(33)* addrspace(66)* getelementptr (%struct.mystruct addrspace(66)* @output, i32 0, i32 1), align 4
	ret void
}

define i32 addrspace(11)* @bar(i32 addrspace(11)* addrspace(22)* addrspace(33)* %x) {
entry:
	%tmp1 = load i32 addrspace(11)* addrspace(22)* addrspace(33)* @y, align 4		; <i32 addrspace(11)* addrspace(22)*> [#uses=2]
	store i32 addrspace(11)* addrspace(22)* %tmp1, i32 addrspace(11)* addrspace(22)* addrspace(33)* %x, align 4
	%tmp5 = load i32 addrspace(11)* addrspace(22)* %tmp1, align 4		; <i32 addrspace(11)*> [#uses=1]
	ret i32 addrspace(11)* %tmp5
}
