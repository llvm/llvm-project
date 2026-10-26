; RUN: opt -verify-each -passes='coro-early,ipsccp,cgscc(coro-split),coro-cleanup,ipsccp,instcombine' -S %s | FileCheck %s
; RUN: opt -verify-each -passes='default<O2>,ipsccp,instcombine' -S %s | FileCheck %s
; RUN: opt -verify-each -passes='thinlto-pre-link<O2>,thinlto<O2>,ipsccp,instcombine' -S %s | FileCheck %s

; The reference from a ramp to its body is initially not a call edge. The body
; calls the ramp recursively, so they are separate call SCCs in one RefSCC.
; Splitting must revisit the ramp and update the graph when that edge becomes
; a direct call. Both direct and mutually recursive entries must stay valid.

declare token @llvm.coro.id.retcon.once(i32, i32, ptr, ptr, ptr, ptr, ptr)
declare ptr @llvm.coro.begin(token, ptr)
declare i1 @llvm.coro.suspend.retcon.i1(...)
declare void @llvm.coro.end(ptr, i1, token)
declare void @llvm.trap()
declare void @prototype(ptr, i1)
declare ptr @malloc(i64)
declare void @free(ptr)

define { ptr, i32 } @recursive(ptr %buffer, i32 %n) noinline presplitcoroutine {
entry:
  %return.slot = alloca { ptr, i32 }, align 8
  %id = call token @llvm.coro.id.retcon.once(i32 16, i32 8, ptr %buffer,
      ptr @prototype, ptr @malloc, ptr @free, ptr %return.slot)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr null)
  %recurse = icmp ne i32 %n, 0
  br i1 %recurse, label %nested, label %yield

nested:
  %next = sub i32 %n, 1
  %storage = alloca [16 x i8], align 8
  %ignored = call { ptr, i32 } @recursive(ptr %storage, i32 %next)
  br label %yield

yield:
  %unwind = call i1 (...) @llvm.coro.suspend.retcon.i1(i32 7)
  call void @llvm.trap()
  unreachable

cleanup:
  call void @llvm.coro.end(ptr %hdl, i1 false, token none)
  %result = load { ptr, i32 }, ptr %return.slot, align 8
  ret { ptr, i32 } %result
}

; CHECK-LABEL: define{{.*}} i32 @caller(
; CHECK: ret i32 7
define i32 @caller(ptr %buffer, i32 %n) {
  %pair = call { ptr, i32 } @recursive(ptr %buffer, i32 %n)
  %yield = extractvalue { ptr, i32 } %pair, 1
  ret i32 %yield
}

define { ptr, i32 } @left(ptr %buffer, i32 %n) noinline presplitcoroutine {
entry:
  %return.slot = alloca { ptr, i32 }, align 8
  %id = call token @llvm.coro.id.retcon.once(i32 16, i32 8, ptr %buffer,
      ptr @prototype, ptr @malloc, ptr @free, ptr %return.slot)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr null)
  %recurse = icmp ne i32 %n, 0
  br i1 %recurse, label %nested, label %yield

nested:
  %next = sub i32 %n, 1
  %storage = alloca [16 x i8], align 8
  %ignored = call { ptr, i32 } @right(ptr %storage, i32 %next)
  br label %yield

yield:
  %unwind = call i1 (...) @llvm.coro.suspend.retcon.i1(i32 7)
  call void @llvm.trap()
  unreachable

cleanup:
  call void @llvm.coro.end(ptr %hdl, i1 false, token none)
  %result = load { ptr, i32 }, ptr %return.slot, align 8
  ret { ptr, i32 } %result
}

define { ptr, i32 } @right(ptr %buffer, i32 %n) noinline presplitcoroutine {
entry:
  %return.slot = alloca { ptr, i32 }, align 8
  %id = call token @llvm.coro.id.retcon.once(i32 16, i32 8, ptr %buffer,
      ptr @prototype, ptr @malloc, ptr @free, ptr %return.slot)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr null)
  %recurse = icmp ne i32 %n, 0
  br i1 %recurse, label %nested, label %yield

nested:
  %next = sub i32 %n, 1
  %storage = alloca [16 x i8], align 8
  %ignored = call { ptr, i32 } @left(ptr %storage, i32 %next)
  br label %yield

yield:
  %unwind = call i1 (...) @llvm.coro.suspend.retcon.i1(i32 7)
  call void @llvm.trap()
  unreachable

cleanup:
  call void @llvm.coro.end(ptr %hdl, i1 false, token none)
  %result = load { ptr, i32 }, ptr %return.slot, align 8
  ret { ptr, i32 } %result
}

; CHECK-LABEL: define{{.*}} i32 @mutual_caller(
; CHECK: ret i32 7
define i32 @mutual_caller(ptr %buffer, i32 %n) {
  %pair = call { ptr, i32 } @left(ptr %buffer, i32 %n)
  %yield = extractvalue { ptr, i32 } %pair, 1
  ret i32 %yield
}
