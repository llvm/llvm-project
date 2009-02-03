; There should be exactly two calls here (memset and malloc), no more.
; RUN: llvm-as < %s | llc -march=alpha | grep jsr | count 2

%typedef.bc_struct = type opaque
declare void @llvm.memset.i64(i8*, i8, i64, i32)

define i1 @l12_l94_bc_divide_endif_2E_3_2E_ce(i32* %tmp.71.reload, i32 %scale2.1.3, i32 %extra.0, %typedef.bc_struct* %n1, %typedef.bc_struct* %n2, i32* %tmp.92.reload, i32 %tmp.94.reload, i32* %tmp.98.reload, i32 %tmp.100.reload, i8** %tmp.112.out, i32* %tmp.157.out, i8** %tmp.158.out) {
newFuncRoot:
        %tmp.120 = add i32 %extra.0, 2          ; <i32> [#uses=1]
        %tmp.122 = add i32 %tmp.120, %tmp.94.reload             ; <i32> [#uses=1]
        %tmp.123 = add i32 %tmp.122, %tmp.100.reload            ; <i32> [#uses=2]
        %tmp.112 = malloc i8, i32 %tmp.123              ; <i8*> [#uses=1]
        %tmp.137 = zext i32 %tmp.123 to i64             ; <i64> [#uses=1]
        tail call void @llvm.memset.i64( i8* %tmp.112, i8 0, i64 %tmp.137, i32 0 )
        ret i1 true
}

