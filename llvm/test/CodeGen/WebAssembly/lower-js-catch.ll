; RUN: opt < %s  -wasm-lower-em-ehsjlj -wasm-enable-sjlj -S | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; Function Attrs: nounwind
define i32 @test_func() #0 {
; CHECK-LABEL: define i32 @test_func()
entry:
  %res = alloca i32, align 4
  %0 = call ptr addrspace(10) @__builtin_wasm_js_catch(ptr nonnull %res) #1
  %1 = load i32, ptr %res, align 4
  %cmp = icmp eq i32 %1, 0
  br i1 %cmp, label %if.then, label %if.else
; CHECK:    entry:
; CHECK-NEXT: %res = alloca i32, align 4
; CHECK-NEXT: %DispatchTarget = alloca i32, align 4
; CHECK-NEXT: %DispatchArgument = alloca ptr, align 4
; CHECK-NEXT: store i32 0, ptr %DispatchTarget, align 4
; CHECK-NEXT: %nullref = call ptr addrspace(10) @llvm.wasm.ref.null.extern()
; CHECK-NEXT: br label %jscatch.dispatch

; CHECK: jscatch.dispatch:
; CHECK-NEXT: %thrown{{[0-9]+}} = phi ptr addrspace(10)
; CHECK-NEXT: %label.phi = phi i32
; CHECK-NEXT: switch i32 %label.phi, label %entry.split [
; CHECK-NEXT: i32 1, label %entry.split.split
; CHECK-NEXT: ]

; CHECK: entry.split:
; CHECK-NEXT: store i32 0, ptr %res, align 4
; CHECK-NEXT: store i32 1, ptr %DispatchTarget, align 4
; CHECK-NEXT: store ptr %res, ptr %DispatchArgument, align 4

; CHECK: entry.split.split:
; CHECK-NEXT: %jscatch.ret = phi ptr addrspace(10)
; CHECK-NEXT: %{{[0-9]+}} = load i32, ptr %res, align 4


if.then:                                          ; preds = %entry
  call void @jsfunc1() #1
  br label %cleanup

; CHECK: if.then:
; CHECK-NEXT: invoke void @jsfunc1()
; CHECK-NEXT: to label %{{.*}} unwind label %catch.dispatch.jserror

if.else:                                          ; preds = %entry
  %call = call i32 @handle_error(ptr addrspace(10) %0) #1
  br label %cleanup

; CHECK: if.else:
; CHECK-NEXT: invoke i32 @handle_error(ptr addrspace(10) %jscatch.ret)
; CHECK-NEXT: to label %{{.*}} unwind label %catch.dispatch.jserror

cleanup:                                          ; preds = %if.then, %if.else
  %retval.0 = phi i32 [ 0, %if.then ], [ %call, %if.else ]
  ret i32 %retval.0

; CHECK: catch.dispatch.jserror:
; CHECK-NEXT: %{{[0-9]+}} = catchswitch within none [label %catch.jserror] unwind to caller

; CHECK: catch.jserror:
; CHECK-NEXT: %{{[0-9]+}} = catchpad within %{{[0-9]+}} []
; CHECK-NEXT: %thrown = call ptr addrspace(10) @llvm.wasm.catch.js()
; CHECK-NEXT: %dispatch.target.value = load i32, ptr %DispatchTarget, align 4
; CHECK-NEXT: %{{[0-9]+}} = icmp eq i32 %dispatch.target.value, 0
; CHECK-NEXT: br i1 %{{[0-9]+}}, label %if.then1, label %if.end

; CHECK: if.then1:
; CHECK-NEXT: call void @llvm.wasm.rethrow() [ "funclet"(token %{{[0-9]+}}) ]
; CHECK-NEXT: unreachable

; CHECK: if.end:
; CHECK-NEXT: %dispatch.argument.value = load ptr, ptr %DispatchArgument, align 4
; CHECK-NEXT: store i32 1, ptr %dispatch.argument.value, align 4
; CHECK-NEXT: catchret from %{{[0-9]+}} to label %jscatch.dispatch
}

declare ptr addrspace(10) @__builtin_wasm_js_catch(ptr) #2
declare void @jsfunc1()
declare i32 @handle_error(ptr addrspace(10))


attributes #0 = { nounwind "target-features"="+exception-handling" }
attributes #1 = { nounwind }
attributes #2 = { "returns_twice" }
