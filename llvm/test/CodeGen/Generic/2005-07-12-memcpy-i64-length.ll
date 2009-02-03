; RUN: llvm-as < %s | llc
; Test that llvm.memcpy works with a i64 length operand on all targets.

declare void @llvm.memcpy.i64(i8*, i8*, i64, i32)

define void @l12_l94_bc_divide_endif_2E_3_2E_ce() {
newFuncRoot:
        tail call void @llvm.memcpy.i64( i8* null, i8* null, i64 0, i32 1 )
        unreachable
}

