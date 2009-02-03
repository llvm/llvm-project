; RUN: llvm-as < %s | llc -march=ppc64 | grep lwzx

        %struct.__db_region = type { %struct.__mutex_t, [4 x i8], %struct.anon, i32, [1 x i32] }
        %struct.__mutex_t = type { i32 }
        %struct.anon = type { i64, i64 }

define void @foo() {
entry:
        %ttype = alloca i32, align 4            ; <i32*> [#uses=1]
        %regs = alloca [1024 x %struct.__db_region], align 16           ; <[1024 x %struct.__db_region]*> [#uses=0]
        %tmp = load i32* %ttype, align 4                ; <i32> [#uses=1]
        %tmp1 = call i32 (...)* @bork( i32 %tmp )               ; <i32> [#uses=0]
        ret void
}

declare i32 @bork(...)
