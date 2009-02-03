; RUN: llvm-as < %s | llc -march=alpha | grep wmb

declare void @llvm.memory.barrier( i1 , i1 , i1 , i1 , i1)

define void @test() {
	call void @llvm.memory.barrier( i1 false, i1 false, i1 false, i1 true , i1 true)
	ret void
}
