; RUN: opt -S %s -passes=verify

declare token @llvm.call.preallocated.setup(i32)
declare ptr @llvm.call.preallocated.arg(token, i32)
declare void @llvm.call.preallocated.teardown(token)

declare i32 @__CxxFrameHandler3(...)

declare void @foo1(ptr preallocated(i32))
declare i64 @foo1_i64(ptr preallocated(i32))
declare void @foo2(ptr preallocated(i32), ptr, ptr preallocated(i32))

declare void @constructor(ptr)

define void @preallocated() {
    %cs = call token @llvm.call.preallocated.setup(i32 1)
    %x = call ptr @llvm.call.preallocated.arg(token %cs, i32 0) preallocated(i32)
    call void @foo1(ptr preallocated(i32) %x) ["preallocated"(token %cs)]
    ret void
}

define void @preallocated_indirect(ptr %f) {
    %cs = call token @llvm.call.preallocated.setup(i32 1)
    %x = call ptr @llvm.call.preallocated.arg(token %cs, i32 0) preallocated(i32)
    call void %f(ptr preallocated(i32) %x) ["preallocated"(token %cs)]
    ret void
}

define void @preallocated_setup_without_call() {
    %cs = call token @llvm.call.preallocated.setup(i32 1)
    %a0 = call ptr @llvm.call.preallocated.arg(token %cs, i32 0) preallocated(i32)
    ret void
}

define void @preallocated_num_args() {
    %cs = call token @llvm.call.preallocated.setup(i32 2)
    %x = call ptr @llvm.call.preallocated.arg(token %cs, i32 0) preallocated(i32)
    %y = call ptr @llvm.call.preallocated.arg(token %cs, i32 1) preallocated(i32)
    %a = inttoptr i32 0 to ptr
    call void @foo2(ptr preallocated(i32) %x, ptr %a, ptr preallocated(i32) %y) ["preallocated"(token %cs)]
    ret void
}

define void @preallocated_musttail(ptr preallocated(i32) %a) {
    musttail call void @foo1(ptr preallocated(i32) %a)
    ret void
}

define i64 @preallocated_musttail_i64(ptr preallocated(i32) %a) {
    %r = musttail call i64 @foo1_i64(ptr preallocated(i32) %a)
    ret i64 %r
}

define void @preallocated_teardown() {
    %cs = call token @llvm.call.preallocated.setup(i32 1)
    call void @llvm.call.preallocated.teardown(token %cs)
    ret void
}

define void @preallocated_teardown_invoke() personality ptr @__CxxFrameHandler3 {
    %cs = call token @llvm.call.preallocated.setup(i32 1)
    %x = call ptr @llvm.call.preallocated.arg(token %cs, i32 0) preallocated(i32)
    invoke void @constructor(ptr %x) to label %conta unwind label %contb
conta:
    call void @foo1(ptr preallocated(i32) %x) ["preallocated"(token %cs)]
    ret void
contb:
    %s = catchswitch within none [label %catch] unwind to caller
catch:
    %p = catchpad within %s []
    call void @llvm.call.preallocated.teardown(token %cs)
    ret void
}
