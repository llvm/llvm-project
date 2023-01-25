; RUN: opt -opaque-pointers=0 < %s -wasm-lower-em-ehsjlj -wasm-enable-sjlj -S | FileCheck %s -DPTR=i32
; RUN: opt -opaque-pointers=0 < %s -wasm-lower-em-ehsjlj -wasm-enable-sjlj -S --mattr=+atomics,+bulk-memory | FileCheck %s -DPTR=i32
; RUN: opt -opaque-pointers=0 < %s -wasm-lower-em-ehsjlj -wasm-enable-sjlj --mtriple=wasm64-unknown-unknown -data-layout="e-m:e-p:64:64-i64:64-n32:64-S128" -S | FileCheck %s -DPTR=i64

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

%struct.__jmp_buf_tag = type { [6 x i32], i32, [32 x i32] }

; These variables are only used in Emscripten EH/SjLj, so they shouldn't be
; generated.
; CHECK-NOT: @__THREW__ =
; CHECK-NOT: @__threwValue =

@global_longjmp_ptr = global void (%struct.__jmp_buf_tag*, i32)* @longjmp, align 4
; CHECK-DAG: @global_longjmp_ptr = global void (%struct.__jmp_buf_tag*, i32)* bitcast (void (i8*, i32)* @__wasm_longjmp to void (%struct.__jmp_buf_tag*, i32)*)

; Test a simple setjmp - longjmp sequence
define void @setjmp_longjmp() {
; CHECK-LABEL: @setjmp_longjmp()
entry:
  %buf = alloca [1 x %struct.__jmp_buf_tag], align 16
  %arraydecay = getelementptr inbounds [1 x %struct.__jmp_buf_tag], [1 x %struct.__jmp_buf_tag]* %buf, i32 0, i32 0
  %call = call i32 @setjmp(%struct.__jmp_buf_tag* %arraydecay) #0
  %arraydecay1 = getelementptr inbounds [1 x %struct.__jmp_buf_tag], [1 x %struct.__jmp_buf_tag]* %buf, i32 0, i32 0
  call void @longjmp(%struct.__jmp_buf_tag* %arraydecay1, i32 1) #1
  unreachable

; CHECK:    entry:
; CHECK-NEXT: %malloccall = tail call i8* @malloc([[PTR]] 40)
; CHECK-NEXT: %setjmpTable = bitcast i8* %malloccall to i32*
; CHECK-NEXT: store i32 0, i32* %setjmpTable, align 4
; CHECK-NEXT: %setjmpTableSize = add i32 4, 0
; CHECK-NEXT: br label %setjmp.dispatch

; CHECK:    setjmp.dispatch:
; CHECK-NEXT: %val10 = phi i32 [ %val, %if.end ], [ undef, %entry ]
; CHECK-NEXT: %buf9 = phi [1 x %struct.__jmp_buf_tag]* [ %buf8, %if.end ], [ undef, %entry ]
; CHECK-NEXT: %setjmpTableSize6 = phi i32 [ %setjmpTableSize7, %if.end ], [ %setjmpTableSize, %entry ]
; CHECK-NEXT: %setjmpTable4 = phi i32* [ %setjmpTable5, %if.end ], [ %setjmpTable, %entry ]
; CHECK-NEXT: %label.phi = phi i32 [ %label, %if.end ], [ -1, %entry ]
; CHECK-NEXT: switch i32 %label.phi, label %entry.split [
; CHECK-NEXT:   i32 1, label %entry.split.split
; CHECK-NEXT: ]

; CHECK:    entry.split:
; CHECK-NEXT: %buf = alloca [1 x %struct.__jmp_buf_tag], align 16
; CHECK-NEXT: %arraydecay = getelementptr inbounds [1 x %struct.__jmp_buf_tag], [1 x %struct.__jmp_buf_tag]* %buf, i32 0, i32 0
; CHECK-NEXT: %setjmpTable1 = call i32* @saveSetjmp(%struct.__jmp_buf_tag* %arraydecay, i32 1, i32* %setjmpTable4, i32 %setjmpTableSize6)
; CHECK-NEXT: %setjmpTableSize2 = call i32 @getTempRet0()
; CHECK-NEXT: br label %entry.split.split

; CHECK:    entry.split.split:
; CHECK-NEXT: %buf8 = phi [1 x %struct.__jmp_buf_tag]* [ %buf9, %setjmp.dispatch ], [ %buf, %entry.split ]
; CHECK-NEXT: %setjmpTableSize7 = phi i32 [ %setjmpTableSize2, %entry.split ], [ %setjmpTableSize6, %setjmp.dispatch ]
; CHECK-NEXT: %setjmpTable5 = phi i32* [ %setjmpTable1, %entry.split ], [ %setjmpTable4, %setjmp.dispatch ]
; CHECK-NEXT: %setjmp.ret = phi i32 [ 0, %entry.split ], [ %val10, %setjmp.dispatch ]
; CHECK-NEXT: %arraydecay1 = getelementptr inbounds [1 x %struct.__jmp_buf_tag], [1 x %struct.__jmp_buf_tag]* %buf8, i32 0, i32 0
; CHECK-NEXT: %env = bitcast %struct.__jmp_buf_tag* %arraydecay1 to i8*
; CHECK-NEXT: invoke void @__wasm_longjmp(i8* %env, i32 1)
; CHECK-NEXT:         to label %.noexc unwind label %catch.dispatch.longjmp

; CHECK:    .noexc:
; CHECK-NEXT: unreachable

; CHECK:    catch.dispatch.longjmp:
; CHECK-NEXT: %0 = catchswitch within none [label %catch.longjmp] unwind to caller

; CHECK:    catch.longjmp:
; CHECK-NEXT: %1 = catchpad within %0 []
; CHECK-NEXT: %thrown = call i8* @llvm.wasm.catch(i32 1)
; CHECK-NEXT: %longjmp.args = bitcast i8* %thrown to { i8*, i32 }*
; CHECK-NEXT: %env_gep = getelementptr { i8*, i32 }, { i8*, i32 }* %longjmp.args, i32 0, i32 0
; CHECK-NEXT: %val_gep = getelementptr { i8*, i32 }, { i8*, i32 }* %longjmp.args, i32 0, i32 1
; CHECK-NEXT: %env3 = load i8*, i8** %env_gep, align {{.*}}
; CHECK-NEXT: %val = load i32, i32* %val_gep, align 4
; CHECK-NEXT: %env.p = bitcast i8* %env3 to [[PTR]]*
; CHECK-NEXT: %setjmp.id = load [[PTR]], [[PTR]]* %env.p, align {{.*}}
; CHECK-NEXT: %label = call i32 @testSetjmp([[PTR]] %setjmp.id, i32* %setjmpTable5, i32 %setjmpTableSize7) [ "funclet"(token %1) ]
; CHECK-NEXT: %2 = icmp eq i32 %label, 0
; CHECK-NEXT: br i1 %2, label %if.then, label %if.end

; CHECK:    if.then:
; CHECK-NEXT: %3 = bitcast i32* %setjmpTable5 to i8*
; CHECK-NEXT: tail call void @free(i8* %3) [ "funclet"(token %1) ]
; CHECK-NEXT: call void @__wasm_longjmp(i8* %env3, i32 %val) [ "funclet"(token %1) ]
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
  %arraydecay = getelementptr inbounds [1 x %struct.__jmp_buf_tag], [1 x %struct.__jmp_buf_tag]* %buf, i32 0, i32 0
  %call = call i32 @setjmp(%struct.__jmp_buf_tag* %arraydecay) #0
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
declare void @take_longjmp(void (%struct.__jmp_buf_tag*, i32)* %arg_ptr)
define void @indirect_longjmp() {
; CHECK-LABEL: @indirect_longjmp
entry:
  %local_longjmp_ptr = alloca void (%struct.__jmp_buf_tag*, i32)*, align 4
  %buf0 = alloca [1 x %struct.__jmp_buf_tag], align 16
  %buf1 = alloca [1 x %struct.__jmp_buf_tag], align 16

  ; Store longjmp in a local variable, load it, and call it
  store void (%struct.__jmp_buf_tag*, i32)* @longjmp, void (%struct.__jmp_buf_tag*, i32)** %local_longjmp_ptr, align 4
  ; CHECK: store void (%struct.__jmp_buf_tag*, i32)* bitcast (void (i8*, i32)* @__wasm_longjmp to void (%struct.__jmp_buf_tag*, i32)*), void (%struct.__jmp_buf_tag*, i32)** %local_longjmp_ptr, align 4
  %longjmp_from_local_ptr = load void (%struct.__jmp_buf_tag*, i32)*, void (%struct.__jmp_buf_tag*, i32)** %local_longjmp_ptr, align 4
  %arraydecay = getelementptr inbounds [1 x %struct.__jmp_buf_tag], [1 x %struct.__jmp_buf_tag]* %buf0, i32 0, i32 0
  call void %longjmp_from_local_ptr(%struct.__jmp_buf_tag* %arraydecay, i32 0)

  ; Load longjmp from a global variable and call it
  %longjmp_from_global_ptr = load void (%struct.__jmp_buf_tag*, i32)*, void (%struct.__jmp_buf_tag*, i32)** @global_longjmp_ptr, align 4
  %arraydecay1 = getelementptr inbounds [1 x %struct.__jmp_buf_tag], [1 x %struct.__jmp_buf_tag]* %buf1, i32 0, i32 0
  call void %longjmp_from_global_ptr(%struct.__jmp_buf_tag* %arraydecay1, i32 0)

  ; Pass longjmp as a function argument. This is a call but longjmp is not a
  ; callee but an argument.
  call void @take_longjmp(void (%struct.__jmp_buf_tag*, i32)* @longjmp)
  ; CHECK: call void @take_longjmp(void (%struct.__jmp_buf_tag*, i32)* bitcast (void (i8*, i32)* @__wasm_longjmp to void (%struct.__jmp_buf_tag*, i32)*))
  ret void
}

; Function Attrs: nounwind
declare void @foo() #2
; The pass removes the 'nounwind' attribute, so there should be no attributes
; CHECK-NOT: declare void @foo #{{.*}}
; Function Attrs: returns_twice
declare i32 @setjmp(%struct.__jmp_buf_tag*) #0
; Function Attrs: noreturn
declare void @longjmp(%struct.__jmp_buf_tag*, i32) #1
declare i32 @__gxx_personality_v0(...)
declare i8* @__cxa_begin_catch(i8*)
declare void @__cxa_end_catch()
declare void @free(i8*)

; JS glue function declarations
; CHECK-DAG: declare i32 @getTempRet0()
; CHECK-DAG: declare i32* @saveSetjmp(%struct.__jmp_buf_tag*, i32, i32*, i32)
; CHECK-DAG: declare i32 @testSetjmp([[PTR]], i32*, i32)
; CHECK-DAG: declare void @__wasm_longjmp(i8*, i32)

attributes #0 = { returns_twice }
attributes #1 = { noreturn }
attributes #2 = { nounwind }
