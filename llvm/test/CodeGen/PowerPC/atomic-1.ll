; RUN: llvm-as < %s | llc -march=ppc32 | grep lwarx  | count 3
; RUN: llvm-as < %s | llc -march=ppc32 | grep stwcx. | count 4

define i32 @exchange_and_add(i32* %mem, i32 %val) nounwind  {
	%tmp = call i32 @llvm.atomic.load.add.i32( i32* %mem, i32 %val )
	ret i32 %tmp
}

define i32 @exchange_and_cmp(i32* %mem) nounwind  {
       	%tmp = call i32 @llvm.atomic.cmp.swap.i32( i32* %mem, i32 0, i32 1 )
	ret i32 %tmp
}

define i32 @exchange(i32* %mem, i32 %val) nounwind  {
	%tmp = call i32 @llvm.atomic.swap.i32( i32* %mem, i32 1 )
	ret i32 %tmp
}

declare i32 @llvm.atomic.load.add.i32(i32*, i32) nounwind 
declare i32 @llvm.atomic.cmp.swap.i32(i32*, i32, i32) nounwind 
declare i32 @llvm.atomic.swap.i32(i32*, i32) nounwind 
