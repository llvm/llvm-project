; RUN: llc < %s -wasm-enable-eh -wasm-use-legacy-eh=false -exception-model=wasm -mattr=+exception-handling -verify-machineinstrs -filetype=obj -o /dev/null
; RUN: llc < %s -mtriple=wasm64-unknown-emscripten -wasm-enable-eh -wasm-use-legacy-eh=false -exception-model=wasm -mattr=+atomics,+bulk-memory,+mutable-globals,+sign-ext,+nontrapping-fptoint,+multivalue,+exception-handling -verify-machineinstrs -filetype=obj -o /dev/null

; Regression test for issue #126916 (one of its subcases): a single
; `cleanuppad` reached as the unwind destination of multiple `invoke`s
; (bb3 and bb7 both unwind to bb11 below). Under the standardized Wasm
; EH codegen path, `placeTryTableMarker` emits two `TRY_TABLE` markers
; for that pad but only one `catch`, so the reverse walk in
; `fixCallUnwindMismatches` underflows `EHPadStack` when it tries to
; match the pop_back against the push. Prior to the fix, this crashed
; with `SmallVector::back()`/`pop_back()` assertions (or a segfault in
; release builds).

target triple = "wasm32-unknown-unknown"

declare i32 @__gxx_wasm_personality_v0(...)
declare ptr @llvm.wasm.get.exception(token)
declare void @f1(ptr, ptr)
declare ptr @f2(ptr, ptr)
declare ptr @f3(ptr, ptr)
declare void @f4(ptr)
declare i32 @f5(ptr)

define i32 @shared_cleanuppad(ptr %0, i1 %1) personality ptr @__gxx_wasm_personality_v0 {
  br i1 %1, label %3, label %14

3:
  invoke void @f1(ptr null, ptr null)
          to label %5 unwind label %11

5:
  %6 = invoke ptr @f2(ptr null, ptr null)
          to label %7 unwind label %9

7:
  %8 = invoke ptr @f3(ptr null, ptr null)
          to label %common.ret unwind label %11

common.ret:
  %common.ret.op = phi i32 [ 0, %23 ], [ 0, %9 ], [ 0, %18 ], [ 0, %26 ], [ 0, %7 ]
  ret i32 %common.ret.op

9:
  %10 = cleanuppad within none []
  br label %common.ret

11:
  %12 = cleanuppad within none []
  cleanupret from %12 unwind to caller

14:
  invoke void @f4(ptr null)
          to label %26 unwind label %16

16:
  %17 = catchswitch within none [label %18] unwind to caller

18:
  %19 = catchpad within %17 [ptr null]
  %20 = tail call ptr @llvm.wasm.get.exception(token %19)
  br label %common.ret

21:
  %22 = catchswitch within none [label %23] unwind to caller

23:
  %24 = catchpad within %22 [ptr null]
  %25 = tail call ptr @llvm.wasm.get.exception(token %24)
  br label %common.ret

26:
  %28 = invoke i32 @f5(ptr null)
          to label %common.ret unwind label %21
}
