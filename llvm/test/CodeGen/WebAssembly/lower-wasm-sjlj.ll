; RUN: opt < %s -wasm-lower-em-ehsjlj -wasm-enable-sjlj -S | FileCheck %s -DPTR=i32
; RUN: opt < %s -wasm-lower-em-ehsjlj -wasm-enable-sjlj -S --mattr=+atomics,+bulk-memory | FileCheck %s -DPTR=i32
; RUN: opt < %s -wasm-lower-em-ehsjlj -wasm-enable-sjlj --mtriple=wasm64-unknown-unknown -data-layout="e-m:e-p:64:64-i64:64-n32:64-S128" -S | FileCheck %s -DPTR=i64

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

%struct.__jmp_buf_tag = type { [6 x i32], i32, [32 x i32] }

; These variables are only used in Emscripten EH/SjLj, so they shouldn't be
; generated.
; CHECK-NOT: @__THREW__ =
; CHECK-NOT: @__threwValue =

@global_longjmp_ptr = global ptr @longjmp, align 4
; CHECK-DAG: @global_longjmp_ptr = global ptr @__wasm_longjmp

; Test a simple setjmp - longjmp sequence
define void @setjmp_longjmp() {
; CHECK-LABEL: @setjmp_longjmp()
entry:
  %buf = alloca [1 x %struct.__jmp_buf_tag], align 16
  %call = call i32 @setjmp(ptr %buf) #0
  call void @longjmp(ptr %buf, i32 1) #1
  unreachable

; CHECK:    entry:
; CHECK-NEXT: %malloccall = tail call ptr @malloc([[PTR]] 40)
; CHECK-NEXT: store i32 0, ptr %malloccall, align 4
; CHECK-NEXT: %setjmpTableSize = add i32 4, 0
; CHECK-NEXT: br label %setjmp.dispatch

; CHECK:    setjmp.dispatch:
; CHECK-NEXT: %val8 = phi i32 [ %val, %if.end ], [ undef, %entry ]
; CHECK-NEXT: %buf7 = phi ptr [ %buf6, %if.end ], [ undef, %entry ]
; CHECK-NEXT: %setjmpTableSize4 = phi i32 [ %setjmpTableSize5, %if.end ], [ %setjmpTableSize, %entry ]
; CHECK-NEXT: %setjmpTable2 = phi ptr [ %setjmpTable3, %if.end ], [ %malloccall, %entry ]
; CHECK-NEXT: %label.phi = phi i32 [ %label, %if.end ], [ -1, %entry ]
; CHECK-NEXT: switch i32 %label.phi, label %entry.split [
; CHECK-NEXT:   i32 1, label %entry.split.split
; CHECK-NEXT: ]

; CHECK:    entry.split:
; CHECK-NEXT: %buf = alloca [1 x %struct.__jmp_buf_tag], align 16
; CHECK-NEXT: %setjmpTable = call ptr @saveSetjmp(ptr %buf, i32 1, ptr %setjmpTable2, i32 %setjmpTableSize4)
; CHECK-NEXT: %setjmpTableSize1 = call i32 @getTempRet0()
; CHECK-NEXT: br label %entry.split.split

; CHECK:    entry.split.split:
; CHECK-NEXT: %buf6 = phi ptr [ %buf7, %setjmp.dispatch ], [ %buf, %entry.split ]
; CHECK-NEXT: %setjmpTableSize5 = phi i32 [ %setjmpTableSize1, %entry.split ], [ %setjmpTableSize4, %setjmp.dispatch ]
; CHECK-NEXT: %setjmpTable3 = phi ptr [ %setjmpTable, %entry.split ], [ %setjmpTable2, %setjmp.dispatch ]
; CHECK-NEXT: %setjmp.ret = phi i32 [ 0, %entry.split ], [ %val8, %setjmp.dispatch ]
; CHECK-NEXT: invoke void @__wasm_longjmp(ptr %buf6, i32 1)
; CHECK-NEXT:         to label %.noexc unwind label %catch.dispatch.longjmp

; CHECK:    .noexc:
; CHECK-NEXT: unreachable

; CHECK:    catch.dispatch.longjmp:
; CHECK-NEXT: %0 = catchswitch within none [label %catch.longjmp] unwind to caller

; CHECK:    catch.longjmp:
; CHECK-NEXT: %1 = catchpad within %0 []
; CHECK-NEXT: %thrown = call ptr @llvm.wasm.catch(i32 1)
; CHECK-NEXT: %env_gep = getelementptr { ptr, i32 }, ptr %thrown, i32 0, i32 0
; CHECK-NEXT: %val_gep = getelementptr { ptr, i32 }, ptr %thrown, i32 0, i32 1
; CHECK-NEXT: %env = load ptr, ptr %env_gep, align {{.*}}
; CHECK-NEXT: %val = load i32, ptr %val_gep, align 4
; CHECK-NEXT: %setjmp.id = load [[PTR]], ptr %env, align {{.*}}
; CHECK-NEXT: %label = call i32 @testSetjmp([[PTR]] %setjmp.id, ptr %setjmpTable3, i32 %setjmpTableSize5) [ "funclet"(token %1) ]
; CHECK-NEXT: %2 = icmp eq i32 %label, 0
; CHECK-NEXT: br i1 %2, label %if.then, label %if.end

; CHECK:    if.then:
; CHECK-NEXT: tail call void @free(ptr %setjmpTable3) [ "funclet"(token %1) ]
; CHECK-NEXT: call void @__wasm_longjmp(ptr %env, i32 %val) [ "funclet"(token %1) ]
; CHECK-NEXT: unreachable

; CHECK:    if.end:
; CHECK-NEXT: catchret from %1 to label %setjmp.dispatch
}

; When there are multiple longjmpable calls after setjmp. This will turn each of
; longjmpable call into an invoke whose unwind destination is
; 'catch.dispatch.longjmp' BB.
define void @setjmp_multiple_longjmpable_calls() {
; CHECK-LABEL: @setjmp_multiple_longjmpable_calls
entry:
  %buf = alloca [1 x %struct.__jmp_buf_tag], align 16
  %call = call i32 @setjmp(ptr %buf) #0
  call void @foo()
  call void @foo()
  ret void

; CHECK: entry.split.split:
; CHECK:   invoke void @foo()
; CHECK:           to label %{{.*}} unwind label %catch.dispatch.longjmp

; CHECK: .noexc:
; CHECK:   invoke void @foo()
; CHECK:           to label %{{.*}} unwind label %catch.dispatch.longjmp
}

; Tests cases where longjmp function pointer is used in other ways than direct
; calls. longjmps should be replaced with (void(*)(jmp_buf*, int))__wasm_longjmp.
declare void @take_longjmp(ptr %arg_ptr)
define void @indirect_longjmp() {
; CHECK-LABEL: @indirect_longjmp
entry:
  %local_longjmp_ptr = alloca ptr, align 4
  %buf0 = alloca [1 x %struct.__jmp_buf_tag], align 16
  %buf1 = alloca [1 x %struct.__jmp_buf_tag], align 16

  ; Store longjmp in a local variable, load it, and call it
  store ptr @longjmp, ptr %local_longjmp_ptr, align 4
  ; CHECK: store ptr @__wasm_longjmp, ptr %local_longjmp_ptr, align 4
  %longjmp_from_local_ptr = load ptr, ptr %local_longjmp_ptr, align 4
  call void %longjmp_from_local_ptr(ptr %buf0, i32 0)

  ; Load longjmp from a global variable and call it
  %longjmp_from_global_ptr = load ptr, ptr @global_longjmp_ptr, align 4
  call void %longjmp_from_global_ptr(ptr %buf1, i32 0)

  ; Pass longjmp as a function argument. This is a call but longjmp is not a
  ; callee but an argument.
  call void @take_longjmp(ptr @longjmp)
  ; CHECK: call void @take_longjmp(ptr @__wasm_longjmp)
  ret void
}

; Function Attrs: nounwind
declare void @foo() #2
; The pass removes the 'nounwind' attribute, so there should be no attributes
; CHECK-NOT: declare void @foo #{{.*}}
; Function Attrs: returns_twice
declare i32 @setjmp(ptr) #0
; Function Attrs: noreturn
declare void @longjmp(ptr, i32) #1
declare i32 @__gxx_personality_v0(...)
declare ptr @__cxa_begin_catch(ptr)
declare void @__cxa_end_catch()
declare void @free(ptr)

; JS glue function declarations
; CHECK-DAG: declare i32 @getTempRet0()
; CHECK-DAG: declare ptr @saveSetjmp(ptr, i32, ptr, i32)
; CHECK-DAG: declare i32 @testSetjmp([[PTR]], ptr, i32)
; CHECK-DAG: declare void @__wasm_longjmp(ptr, i32)

attributes #0 = { returns_twice }
attributes #1 = { noreturn }
attributes #2 = { nounwind }
