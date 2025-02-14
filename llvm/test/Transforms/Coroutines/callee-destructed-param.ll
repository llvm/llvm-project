; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S | FileCheck %s

; Inspired by coro-param-copy.ll and the following:
;
;   $ clang++ -std=c++20 -S -emit-llvm -g0 -fno-exceptions \
;         -mllvm -disable-llvm-optzns -fno-discard-value-names a.cc -o -
;
;   #include <coroutine>
;
;   class BasicCoroutine {
;   public:
;     struct Promise {
;       BasicCoroutine get_return_object();
;       void unhandled_exception() noexcept;
;       void return_void() noexcept;
;       std::suspend_never initial_suspend() noexcept;
;       std::suspend_never final_suspend() noexcept;
;     };
;     using promise_type = Promise;
;   };
;
;   struct [[clang::trivial_abi]] Trivial {
;     Trivial(int x) : x(x) {}
;     ~Trivial();
;     int x;
;   };
;
;   BasicCoroutine coro(Trivial t) {
;     co_return;
;   }
;
;
; Check that even though %x.local may escape via use() in the beginning of @f,
; it is not put in the corountine frame, since %x.local is used after
; @llvm.coro.end, at which point the coroutine frame may have been deallocated.
;
; In the program above, a move constructor (or just memcpy) is invoked to copy
; t to a coroutine-local alloca. At the end of the function, t's destructor is
; called because of trivial_abi. At that point, t must not have been stored in
; the coro frame.

; The frame should not contain an i64.
; CHECK: %f.Frame = type { ptr, ptr, i1 }

; Check that we have both uses of %x.local (and they're not using the frame).
; CHECK-LABEL: define ptr @f(i64 %x)
; CHECK: call void @use(ptr %x.local)
; CHECK: call void @use(ptr %x.local)


define ptr @f(i64 %x) presplitcoroutine {
entry:
  %x.local = alloca i64
  store i64 %x, ptr %x.local
  br label %coro.alloc

coro.alloc:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @myAlloc(i32 %size)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %alloc)
  call void @use(ptr %x.local)
  %0 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %0, label %suspend [i8 0, label %resume
                                i8 1, label %cleanup]
resume:
  br label %cleanup

cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend

suspend:
  call i1 @llvm.coro.end(ptr %hdl, i1 0, token none)
  call void @use(ptr %x.local)  ; It better not be on the frame, that's gone.
  ret ptr %hdl
}

declare ptr @llvm.coro.free(token, ptr)
declare i32 @llvm.coro.size.i32()
declare i8  @llvm.coro.suspend(token, i1)
declare void @llvm.coro.resume(ptr)
declare void @llvm.coro.destroy(ptr)

declare token @llvm.coro.id(i32, ptr, ptr, ptr)
declare i1 @llvm.coro.alloc(token)
declare ptr @llvm.coro.begin(token, ptr)
declare i1 @llvm.coro.end(ptr, i1, token)


declare noalias ptr @myAlloc(i32)
declare void @use(ptr)
declare void @free(ptr)
