; RUN: llvm-as < %s | llc -march=ppc64 | grep ldarx  | count 3
; RUN: llvm-as < %s | llc -march=ppc64 | grep stdcx. | count 4

define i64 @exchange_and_add(i64* %mem, i64 %val) nounwind  {
	%tmp = call i64 @llvm.atomic.load.add.i64( i64* %mem, i64 %val )
	ret i64 %tmp
}

define i64 @exchange_and_cmp(i64* %mem) nounwind  {
       	%tmp = call i64 @llvm.atomic.cmp.swap.i64( i64* %mem, i64 0, i64 1 )
	ret i64 %tmp
}

define i64 @exchange(i64* %mem, i64 %val) nounwind  {
	%tmp = call i64 @llvm.atomic.swap.i64( i64* %mem, i64 1 )
	ret i64 %tmp
}

declare i64 @llvm.atomic.load.add.i64(i64*, i64) nounwind 
declare i64 @llvm.atomic.cmp.swap.i64(i64*, i64, i64) nounwind 
declare i64 @llvm.atomic.swap.i64(i64*, i64) nounwind 
