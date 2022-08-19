; RUN: opt < %s -wasm-lower-em-ehsjlj -enable-emscripten-sjlj -S | FileCheck %s -DPTR=i32
; RUN: opt < %s -wasm-lower-em-ehsjlj -enable-emscripten-sjlj -S --mattr=+atomics,+bulk-memory | FileCheck %s -DPTR=i32
; RUN: opt < %s -wasm-lower-em-ehsjlj -enable-emscripten-sjlj --mtriple=wasm64-unknown-unknown -data-layout="e-m:e-p:64:64-i64:64-n32:64-S128" -S | FileCheck %s -DPTR=i64

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

%struct.__jmp_buf_tag = type { [6 x i32], i32, [32 x i32] }

@global_var = global i32 0, align 4
@global_longjmp_ptr = global void (%struct.__jmp_buf_tag*, i32)* @longjmp, align 4
; CHECK-DAG: @__THREW__ = external thread_local global [[PTR]]
; CHECK-DAG: @__threwValue = external thread_local global i32
; CHECK-DAG: @global_longjmp_ptr = global void (%struct.__jmp_buf_tag*, i32)* bitcast (void ([[PTR]], i32)* @emscripten_longjmp to void (%struct.__jmp_buf_tag*, i32)*)

; Test a simple setjmp - longjmp sequence
define void @setjmp_longjmp() {
; CHECK-LABEL: @setjmp_longjmp
entry:
  %buf = alloca [1 x %struct.__jmp_buf_tag], align 16
  %arraydecay = getelementptr inbounds [1 x %struct.__jmp_buf_tag], [1 x %struct.__jmp_buf_tag]* %buf, i32 0, i32 0
  %call = call i32 @setjmp(%struct.__jmp_buf_tag* %arraydecay) #0
  %arraydecay1 = getelementptr inbounds [1 x %struct.__jmp_buf_tag], [1 x %struct.__jmp_buf_tag]* %buf, i32 0, i32 0
  call void @longjmp(%struct.__jmp_buf_tag* %arraydecay1, i32 1) #1
  unreachable
; CHECK: entry:
; CHECK-NEXT: %[[MALLOCCALL:.*]] = tail call i8* @malloc([[PTR]] 40)
; CHECK-NEXT: %[[SETJMP_TABLE:.*]] = bitcast i8* %[[MALLOCCALL]] to i32*
; CHECK-NEXT: store i32 0, i32* %[[SETJMP_TABLE]]
; CHECK-NEXT: %[[SETJMP_TABLE_SIZE:.*]] = add i32 4, 0
; CHECK-NEXT: br label %entry.split

; CHECK: entry.split
; CHECK-NEXT: %[[BUF:.*]] = alloca [1 x %struct.__jmp_buf_tag]
; CHECK-NEXT: %[[ARRAYDECAY:.*]] = getelementptr inbounds [1 x %struct.__jmp_buf_tag], [1 x %struct.__jmp_buf_tag]* %[[BUF]], i32 0, i32 0
; CHECK-NEXT: %[[SETJMP_TABLE1:.*]] = call i32* @saveSetjmp(%struct.__jmp_buf_tag* %[[ARRAYDECAY]], i32 1, i32* %[[SETJMP_TABLE]], i32 %[[SETJMP_TABLE_SIZE]])
; CHECK-NEXT: %[[SETJMP_TABLE_SIZE1:.*]] = call i32 @getTempRet0()
; CHECK-NEXT: br label %entry.split.split

; CHECK: entry.split.split:
; CHECK-NEXT: phi i32 [ 0, %entry.split ], [ %[[LONGJMP_RESULT:.*]], %if.end ]
; CHECK-NEXT: %[[ARRAYDECAY1:.*]] = getelementptr inbounds [1 x %struct.__jmp_buf_tag], [1 x %struct.__jmp_buf_tag]* %[[BUF]], i32 0, i32 0
; CHECK-NEXT: %[[JMPBUF:.*]] = ptrtoint %struct.__jmp_buf_tag* %[[ARRAYDECAY1]] to [[PTR]]
; CHECK-NEXT: store [[PTR]] 0, [[PTR]]* @__THREW__
; CHECK-NEXT: call cc{{.*}} void @__invoke_void_[[PTR]]_i32(void ([[PTR]], i32)* @emscripten_longjmp, [[PTR]] %[[JMPBUF]], i32 1)
; CHECK-NEXT: %[[__THREW__VAL:.*]] = load [[PTR]], [[PTR]]* @__THREW__
; CHECK-NEXT: store [[PTR]] 0, [[PTR]]* @__THREW__
; CHECK-NEXT: %[[CMP0:.*]] = icmp ne [[PTR]] %__THREW__.val, 0
; CHECK-NEXT: %[[THREWVALUE_VAL:.*]] = load i32, i32* @__threwValue
; CHECK-NEXT: %[[CMP1:.*]] = icmp ne i32 %[[THREWVALUE_VAL]], 0
; CHECK-NEXT: %[[CMP:.*]] = and i1 %[[CMP0]], %[[CMP1]]
; CHECK-NEXT: br i1 %[[CMP]], label %if.then1, label %if.else1

; CHECK: entry.split.split.split:
; CHECK-NEXT: unreachable

; CHECK: if.then1:
; CHECK-NEXT: %[[__THREW__VAL_P:.*]] = inttoptr [[PTR]] %[[__THREW__VAL]] to [[PTR]]*
; CHECK-NEXT: %[[__THREW__VAL_P_LOADED:.*]] = load [[PTR]], [[PTR]]* %[[__THREW__VAL_P]]
; CHECK-NEXT: %[[LABEL:.*]] = call i32 @testSetjmp([[PTR]] %[[__THREW__VAL_P_LOADED]], i32* %[[SETJMP_TABLE1]], i32 %[[SETJMP_TABLE_SIZE1]])
; CHECK-NEXT: %[[CMP:.*]] = icmp eq i32 %[[LABEL]], 0
; CHECK-NEXT: br i1 %[[CMP]], label %call.em.longjmp, label %if.end2

; CHECK: if.else1:
; CHECK-NEXT: br label %if.end

; CHECK: if.end:
; CHECK-NEXT: %[[LABEL_PHI:.*]] = phi i32 [ %[[LABEL:.*]], %if.end2 ], [ -1, %if.else1 ]
; CHECK-NEXT: %[[LONGJMP_RESULT]] = call i32 @getTempRet0()
; CHECK-NEXT: switch i32 %[[LABEL_PHI]], label %entry.split.split.split [
; CHECK-NEXT:   i32 1, label %entry.split.split
; CHECK-NEXT: ]

; CHECK: call.em.longjmp:
; CHECK-NEXT: %threw.phi = phi [[PTR]] [ %[[__THREW__VAL]], %if.then1 ]
; CHECK-NEXT: %threwvalue.phi = phi i32 [ %[[THREWVALUE_VAL]], %if.then1 ]
; CHECK-NEXT: %{{.*}} = bitcast i32* %[[SETJMP_TABLE1]] to i8*
; CHECK-NEXT: tail call void @free(i8* %{{.*}})
; CHECK-NEXT: call void @emscripten_longjmp([[PTR]] %threw.phi, i32 %threwvalue.phi)
; CHECK-NEXT: unreachable

; CHECK: if.end2:
; CHECK-NEXT: call void  @setTempRet0(i32 %[[THREWVALUE_VAL]])
; CHECK-NEXT: br label %if.end
}

; Test a case of a function call (which is not longjmp) after a setjmp
define void @setjmp_longjmpable_call() {
; CHECK-LABEL: @setjmp_longjmpable_call
entry:
  %buf = alloca [1 x %struct.__jmp_buf_tag], align 16
  %arraydecay = getelementptr inbounds [1 x %struct.__jmp_buf_tag], [1 x %struct.__jmp_buf_tag]* %buf, i32 0, i32 0
  %call = call i32 @setjmp(%struct.__jmp_buf_tag* %arraydecay) #0
  call void @foo()
  ret void
; CHECK: entry:
; CHECK: %[[SETJMP_TABLE:.*]] = call i32* @saveSetjmp(

; CHECK: entry.split.split:
; CHECK: @__invoke_void(void ()* @foo)

; CHECK: entry.split.split.split:
; CHECK-NEXT: %[[BUF:.*]] = bitcast i32* %[[SETJMP_TABLE]] to i8*
; CHECK-NEXT: tail call void @free(i8* %[[BUF]])
; CHECK-NEXT: ret void
}

; When there are multiple longjmpable calls after setjmp. In this test we
; specifically check if 'call.em.longjmp' BB, which rethrows longjmps by calling
; emscripten_longjmp for ones that are not for this function's setjmp, is
; correctly created for multiple predecessors.
define void @setjmp_multiple_longjmpable_calls() {
; CHECK-LABEL: @setjmp_multiple_longjmpable_calls
entry:
  %buf = alloca [1 x %struct.__jmp_buf_tag], align 16
  %arraydecay = getelementptr inbounds [1 x %struct.__jmp_buf_tag], [1 x %struct.__jmp_buf_tag]* %buf, i32 0, i32 0
  %call = call i32 @setjmp(%struct.__jmp_buf_tag* %arraydecay) #0
  call void @foo()
  call void @foo()
  ret void
; CHECK: call.em.longjmp:
; CHECK-NEXT:  %threw.phi = phi [[PTR]] [ %__THREW__.val, %if.then1 ], [ %__THREW__.val4, %if.then15 ]
; CHECK-NEXT:  %threwvalue.phi = phi i32 [ %__threwValue.val, %if.then1 ], [ %__threwValue.val8, %if.then15 ]
; CHECK-NEXT: %{{.*}} = bitcast i32* %[[SETJMP_TABLE1]] to i8*
; CHECK-NEXT: tail call void @free(i8* %{{.*}})
; CHECK-NEXT: call void @emscripten_longjmp([[PTR]] %threw.phi, i32 %threwvalue.phi)
; CHECK-NEXT: unreachable
}

; Test a case where a function has a setjmp call but no other calls that can
; longjmp. We don't need to do any transformation in this case.
define i32 @setjmp_only(i8* %ptr) {
; CHECK-LABEL: @setjmp_only
entry:
  %buf = alloca [1 x %struct.__jmp_buf_tag], align 16
  %arraydecay = getelementptr inbounds [1 x %struct.__jmp_buf_tag], [1 x %struct.__jmp_buf_tag]* %buf, i32 0, i32 0
  %call = call i32 @setjmp(%struct.__jmp_buf_tag* %arraydecay) #0
  ; free cannot longjmp
  call void @free(i8* %ptr)
  ret i32 %call
; CHECK: entry:
; CHECK-NOT: @malloc
; CHECK-NOT: %setjmpTable
; CHECK-NOT: @saveSetjmp
; CHECK-NOT: @testSetjmp
; The remaining setjmp call is converted to constant 0, because setjmp returns 0
; when called directly.
; CHECK: ret i32 0
}

; Test SSA validity
define void @ssa(i32 %n) {
; CHECK-LABEL: @ssa
entry:
  %buf = alloca [1 x %struct.__jmp_buf_tag], align 16
  %cmp = icmp sgt i32 %n, 5
  br i1 %cmp, label %if.then, label %if.end
; CHECK: entry:
; CHECK: %[[SETJMP_TABLE0:.*]] = bitcast i8*
; CHECK: %[[SETJMP_TABLE_SIZE0:.*]] = add i32 4, 0

if.then:                                          ; preds = %entry
  %0 = load i32, i32* @global_var, align 4
  %arraydecay = getelementptr inbounds [1 x %struct.__jmp_buf_tag], [1 x %struct.__jmp_buf_tag]* %buf, i32 0, i32 0
  %call = call i32 @setjmp(%struct.__jmp_buf_tag* %arraydecay) #0
  store i32 %0, i32* @global_var, align 4
  br label %if.end
; CHECK: if.then:
; CHECK: %[[VAR0:.*]] = load i32, i32* @global_var, align 4
; CHECK: %[[SETJMP_TABLE1:.*]] = call i32* @saveSetjmp(
; CHECK-NEXT: %[[SETJMP_TABLE_SIZE1:.*]] = call i32 @getTempRet0()

; CHECK: if.then.split:
; CHECK: %[[VAR1:.*]] = phi i32 [ %[[VAR2:.*]], %if.end3 ], [ %[[VAR0]], %if.then ]
; CHECK: %[[SETJMP_TABLE_SIZE2:.*]] = phi i32 [ %[[SETJMP_TABLE_SIZE1]], %if.then ], [ %[[SETJMP_TABLE_SIZE3:.*]], %if.end3 ]
; CHECK: %[[SETJMP_TABLE2:.*]] = phi i32* [ %[[SETJMP_TABLE1]], %if.then ], [ %[[SETJMP_TABLE3:.*]], %if.end3 ]
; CHECK: store i32 %[[VAR1]], i32* @global_var, align 4

if.end:                                           ; preds = %if.then, %entry
  %arraydecay1 = getelementptr inbounds [1 x %struct.__jmp_buf_tag], [1 x %struct.__jmp_buf_tag]* %buf, i32 0, i32 0
  call void @longjmp(%struct.__jmp_buf_tag* %arraydecay1, i32 5) #1
  unreachable
; CHECK: if.end:
; CHECK: %[[VAR2]] = phi i32 [ %[[VAR1]], %if.then.split ], [ undef, %entry.split ]
; CHECK: %[[SETJMP_TABLE_SIZE3]] = phi i32 [ %[[SETJMP_TABLE_SIZE2]], %if.then.split ], [ %[[SETJMP_TABLE_SIZE0]], %entry.split ]
; CHECK: %[[SETJMP_TABLE3]] = phi i32* [ %[[SETJMP_TABLE2]], %if.then.split ], [ %[[SETJMP_TABLE0]], %entry.split ]
}

; Test a case when a function only calls other functions that are neither setjmp nor longjmp
define void @other_func_only() {
; CHECK-LABEL: @other_func_only
entry:
  call void @foo()
  ret void
; CHECK: call void @foo()
}

; Test inline asm handling
define void @inline_asm() {
; CHECK-LABEL: @inline_asm
entry:
  %env = alloca [1 x %struct.__jmp_buf_tag], align 16
  %arraydecay = getelementptr inbounds [1 x %struct.__jmp_buf_tag], [1 x %struct.__jmp_buf_tag]* %env, i32 0, i32 0
  %call = call i32 @setjmp(%struct.__jmp_buf_tag* %arraydecay) #4
; Inline assembly should not generate __invoke wrappers.
; Doing so would fail as inline assembly cannot be passed as a function pointer.
; CHECK: call void asm sideeffect "", ""()
; CHECK-NOT: __invoke_void
  call void asm sideeffect "", ""()
  ret void
}

; Test that the allocsize attribute is being transformed properly
declare i8 *@allocator(i32, %struct.__jmp_buf_tag*) #3
define i8 *@allocsize() {
; CHECK-LABEL: @allocsize
entry:
  %buf = alloca [1 x %struct.__jmp_buf_tag], align 16
  %arraydecay = getelementptr inbounds [1 x %struct.__jmp_buf_tag], [1 x %struct.__jmp_buf_tag]* %buf, i32 0, i32 0
  %call = call i32 @setjmp(%struct.__jmp_buf_tag* %arraydecay) #0
; CHECK: call cc{{.*}} i8* @"__invoke_i8*_i32_%struct.__jmp_buf_tag*"([[ARGS:.*]]) #[[ALLOCSIZE_ATTR:[0-9]+]]
  %alloc = call i8* @allocator(i32 20, %struct.__jmp_buf_tag* %arraydecay) #3
  ret i8 *%alloc
}

; Test a case when a function only calls longjmp and not setjmp
@buffer = global [1 x %struct.__jmp_buf_tag] zeroinitializer, align 16
define void @longjmp_only() {
; CHECK-LABEL: @longjmp_only
entry:
  ; CHECK: call void @emscripten_longjmp
  call void @longjmp(%struct.__jmp_buf_tag* getelementptr inbounds ([1 x %struct.__jmp_buf_tag], [1 x %struct.__jmp_buf_tag]* @buffer, i32 0, i32 0), i32 1) #1
  unreachable
}

; Tests if SSA rewrite works when a use and its def are within the same BB.
define void @ssa_rewite_in_same_bb() {
; CHECK-LABEL: @ssa_rewite_in_same_bb
entry:
  call void @foo()
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  ; CHECK: %{{.*}} = phi i32 [ %var[[VARNO:.*]], %for.inc.split ]
  %0 = phi i32 [ %var, %for.inc ], [ undef, %entry ]
  %var = add i32 0, 0
  br label %for.inc

for.inc:                                          ; preds = %for.cond
  %call5 = call i32 @setjmp(%struct.__jmp_buf_tag* undef) #0
  br label %for.cond

; CHECK: for.inc.split:
  ; CHECK: %var[[VARNO]] = phi i32 [ undef, %if.end ], [ %var, %for.inc ]
}

; Tests cases where longjmp function pointer is used in other ways than direct
; calls. longjmps should be replaced with
; (void(*)(jmp_buf*, int))emscripten_longjmp.
declare void @take_longjmp(void (%struct.__jmp_buf_tag*, i32)* %arg_ptr)
define void @indirect_longjmp() {
; CHECK-LABEL: @indirect_longjmp
entry:
  %local_longjmp_ptr = alloca void (%struct.__jmp_buf_tag*, i32)*, align 4
  %buf0 = alloca [1 x %struct.__jmp_buf_tag], align 16
  %buf1 = alloca [1 x %struct.__jmp_buf_tag], align 16

  ; Store longjmp in a local variable, load it, and call it
  store void (%struct.__jmp_buf_tag*, i32)* @longjmp, void (%struct.__jmp_buf_tag*, i32)** %local_longjmp_ptr, align 4
  ; CHECK: store void (%struct.__jmp_buf_tag*, i32)* bitcast (void ([[PTR]], i32)* @emscripten_longjmp to void (%struct.__jmp_buf_tag*, i32)*), void (%struct.__jmp_buf_tag*, i32)** %local_longjmp_ptr, align 4
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
  ; CHECK: call void @take_longjmp(void (%struct.__jmp_buf_tag*, i32)* bitcast (void ([[PTR]], i32)* @emscripten_longjmp to void (%struct.__jmp_buf_tag*, i32)*))
  ret void
}

; Test if _setjmp and _longjmp calls are treated in the same way as setjmp and
; longjmp
define void @_setjmp__longjmp() {
; CHECK-LABEL: @_setjmp__longjmp
; These calls should have been transformed away
; CHECK-NOT: call i32 @_setjmp
; CHECK-NOT: call void @_longjmp
entry:
  %buf = alloca [1 x %struct.__jmp_buf_tag], align 16
  %arraydecay = getelementptr inbounds [1 x %struct.__jmp_buf_tag], [1 x %struct.__jmp_buf_tag]* %buf, i32 0, i32 0
  %call = call i32 @_setjmp(%struct.__jmp_buf_tag* %arraydecay) #0
  %arraydecay1 = getelementptr inbounds [1 x %struct.__jmp_buf_tag], [1 x %struct.__jmp_buf_tag]* %buf, i32 0, i32 0
  call void @_longjmp(%struct.__jmp_buf_tag* %arraydecay1, i32 1) #1
  unreachable
}

; Function Attrs: nounwind
declare void @foo() #2
; Function Attrs: returns_twice
declare i32 @setjmp(%struct.__jmp_buf_tag*) #0
declare i32 @_setjmp(%struct.__jmp_buf_tag*) #0
; Function Attrs: noreturn
declare void @longjmp(%struct.__jmp_buf_tag*, i32) #1
declare void @_longjmp(%struct.__jmp_buf_tag*, i32) #1
declare i32 @__gxx_personality_v0(...)
declare i8* @__cxa_begin_catch(i8*)
declare void @__cxa_end_catch()
declare void @free(i8*)

; JS glue functions and invoke wrappers declaration
; CHECK-DAG: declare i32 @getTempRet0()
; CHECK-DAG: declare void @setTempRet0(i32)
; CHECK-DAG: declare i32* @saveSetjmp(%struct.__jmp_buf_tag*, i32, i32*, i32)
; CHECK-DAG: declare i32 @testSetjmp([[PTR]], i32*, i32)
; CHECK-DAG: declare void @emscripten_longjmp([[PTR]], i32)
; CHECK-DAG: declare void @__invoke_void(void ()*)

attributes #0 = { returns_twice }
attributes #1 = { noreturn }
attributes #2 = { nounwind }
attributes #3 = { allocsize(0) }
; CHECK-DAG: attributes #{{[0-9]+}} = { nounwind "wasm-import-module"="env" "wasm-import-name"="getTempRet0" }
; CHECK-DAG: attributes #{{[0-9]+}} = { nounwind "wasm-import-module"="env" "wasm-import-name"="setTempRet0" }
; CHECK-DAG: attributes #{{[0-9]+}} = { "wasm-import-module"="env" "wasm-import-name"="__invoke_void" }
; CHECK-DAG: attributes #{{[0-9]+}} = { "wasm-import-module"="env" "wasm-import-name"="saveSetjmp" }
; CHECK-DAG: attributes #{{[0-9]+}} = { "wasm-import-module"="env" "wasm-import-name"="testSetjmp" }
; CHECK-DAG: attributes #{{[0-9]+}} = { noreturn "wasm-import-module"="env" "wasm-import-name"="emscripten_longjmp" }
; CHECK-DAG: attributes #{{[0-9]+}} = { "wasm-import-module"="env" "wasm-import-name"="__invoke_i8*_i32_%struct.__jmp_buf_tag*" }
; CHECK-DAG: attributes #[[ALLOCSIZE_ATTR]] = { allocsize(1) }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !DIFile(filename: "lower-em-sjlj.c", directory: "test")
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1)
!3 = distinct !DISubprogram(name: "setjmp_debug_info", unit:!2, file: !1, line: 1)
!4 = !DILocation(line:2, scope: !3)
!5 = !DILocation(line:3, scope: !3)
!6 = !DILocation(line:4, scope: !3)
!7 = !DILocation(line:5, scope: !3)
!8 = !DILocation(line:6, scope: !3)
