; RUN: llvm-as %s -o - | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

declare void @"foo"()


; foo test basic arith operations
define void @"foo"() {
	%t1 = malloc i31, i32 4
        %t2 = malloc i31, i32 7, align 1024
        %t3 = malloc [4 x i15]

        %idx = getelementptr [4 x i15]* %t3, i64 0, i64 2
        store i15 -123, i15* %idx

        free [4 x i15]* %t3
        free i31* %t2
        free i31* %t1
        
        %t4 = alloca i12, i32 100
        free i12* %t4

        %t5 = alloca i31
        store i31 -123, i31* %t5

        free i31* %t5
	ret void
}
