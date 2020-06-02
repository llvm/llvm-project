; RUN: opt -S %s -verify

declare token @llvm.call.preallocated.setup(i32)
declare i8* @llvm.call.preallocated.arg(token, i32)

declare void @foo1(i32* preallocated(i32))
declare i64 @foo1_i64(i32* preallocated(i32))
declare void @foo2(i32* preallocated(i32), i32*, i32* preallocated(i32))

define void @preallocated() {
    %cs = call token @llvm.call.preallocated.setup(i32 1)
    %x = call i8* @llvm.call.preallocated.arg(token %cs, i32 0) preallocated(i32)
    %y = bitcast i8* %x to i32*
    call void @foo1(i32* preallocated(i32) %y) ["preallocated"(token %cs)]
    ret void
}

define void @preallocated_indirect(void (i32*)* %f) {
    %cs = call token @llvm.call.preallocated.setup(i32 1)
    %x = call i8* @llvm.call.preallocated.arg(token %cs, i32 0) preallocated(i32)
    %y = bitcast i8* %x to i32*
    call void %f(i32* preallocated(i32) %y) ["preallocated"(token %cs)]
    ret void
}

define void @preallocated_setup_without_call() {
    %cs = call token @llvm.call.preallocated.setup(i32 1)
    %a0 = call i8* @llvm.call.preallocated.arg(token %cs, i32 0) preallocated(i32)
    ret void
}

define void @preallocated_num_args() {
    %cs = call token @llvm.call.preallocated.setup(i32 2)
    %x = call i8* @llvm.call.preallocated.arg(token %cs, i32 0) preallocated(i32)
    %x1 = bitcast i8* %x to i32*
    %y = call i8* @llvm.call.preallocated.arg(token %cs, i32 1) preallocated(i32)
    %y1 = bitcast i8* %y to i32*
    %a = inttoptr i32 0 to i32*
    call void @foo2(i32* preallocated(i32) %x1, i32* %a, i32* preallocated(i32) %y1) ["preallocated"(token %cs)]
    ret void
}

define void @preallocate_musttail(i32* preallocated(i32) %a) {
    musttail call void @foo1(i32* preallocated(i32) %a)
    ret void
}

define i64 @preallocate_musttail_i64(i32* preallocated(i32) %a) {
    %r = musttail call i64 @foo1_i64(i32* preallocated(i32) %a)
    ret i64 %r
}
