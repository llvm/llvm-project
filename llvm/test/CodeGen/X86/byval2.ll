; RUN: llvm-as < %s | llc -march=x86-64 | grep rep.movsq | count 2
; RUN: llvm-as < %s | llc -march=x86    | grep rep.movsl | count 2

%struct.s = type { i64, i64, i64, i64, i64, i64, i64, i64,
                   i64, i64, i64, i64, i64, i64, i64, i64,
                   i64 }

define void @g(i64 %a, i64 %b, i64 %c) {
entry:
	%d = alloca %struct.s, align 16
	%tmp = getelementptr %struct.s* %d, i32 0, i32 0
	store i64 %a, i64* %tmp, align 16
	%tmp2 = getelementptr %struct.s* %d, i32 0, i32 1
	store i64 %b, i64* %tmp2, align 16
	%tmp4 = getelementptr %struct.s* %d, i32 0, i32 2
	store i64 %c, i64* %tmp4, align 16
	call void @f( %struct.s* %d byval)
	call void @f( %struct.s* %d byval)
	ret void
}

declare void @f(%struct.s* byval)
