; RUN: opt < %s -wasm-lower-em-ehsjlj -S | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

%struct.__jmp_buf_tag = type { [6 x i32], i32, [32 x i32] }

@global_var = hidden global i32 0, align 4
; CHECK-DAG: @[[__THREW__:__THREW__.*]] = global i32 0
; CHECK-DAG: @[[THREWVALUE:__threwValue.*]] = global i32 0
; CHECK-DAG: @[[TEMPRET0:__tempRet0.*]] = global i32 0

; Test a simple setjmp - longjmp sequence
define hidden void @setjmp_longjmp() {
; CHECK-LABEL: @setjmp_longjmp
entry:
  %buf = alloca [1 x %struct.__jmp_buf_tag], align 16
  %arraydecay = getelementptr inbounds [1 x %struct.__jmp_buf_tag], [1 x %struct.__jmp_buf_tag]* %buf, i32 0, i32 0
  %call = call i32 @setjmp(%struct.__jmp_buf_tag* %arraydecay) #0
  %arraydecay1 = getelementptr inbounds [1 x %struct.__jmp_buf_tag], [1 x %struct.__jmp_buf_tag]* %buf, i32 0, i32 0
  call void @longjmp(%struct.__jmp_buf_tag* %arraydecay1, i32 1) #1
  unreachable
; CHECK: entry:
; CHECK-NEXT: %[[MALLOCCALL:.*]] = tail call i8* @malloc(i32 40)
; CHECK-NEXT: %[[SETJMP_TABLE:.*]] = bitcast i8* %[[MALLOCCALL]] to i32*
; CHECK-NEXT: store i32 0, i32* %[[SETJMP_TABLE]]
; CHECK-NEXT: %[[SETJMP_TABLE_SIZE:.*]] = add i32 4, 0
; CHECK-NEXT: %[[BUF:.*]] = alloca [1 x %struct.__jmp_buf_tag]
; CHECK-NEXT: %[[ARRAYDECAY:.*]] = getelementptr inbounds [1 x %struct.__jmp_buf_tag], [1 x %struct.__jmp_buf_tag]* %[[BUF]], i32 0, i32 0
; CHECK-NEXT: %[[SETJMP_TABLE1:.*]] = call i32* @saveSetjmp(%struct.__jmp_buf_tag* %[[ARRAYDECAY]], i32 1, i32* %[[SETJMP_TABLE]], i32 %[[SETJMP_TABLE_SIZE]])
; CHECK-NEXT: %[[SETJMP_TABLE_SIZE1:.*]] = load i32, i32* @[[TEMPRET0]]
; CHECK-NEXT: br label %entry.split

; CHECK: entry.split:
; CHECK-NEXT: phi i32 [ 0, %entry ], [ %[[LONGJMP_RESULT:.*]], %if.end ]
; CHECK-NEXT: %[[ARRAYDECAY1:.*]] = getelementptr inbounds [1 x %struct.__jmp_buf_tag], [1 x %struct.__jmp_buf_tag]* %[[BUF]], i32 0, i32 0
; CHECK-NEXT: store i32 0, i32* @[[__THREW__]]
; CHECK-NEXT: call void @"__invoke_void_%struct.__jmp_buf_tag*_i32"(void (%struct.__jmp_buf_tag*, i32)* @emscripten_longjmp_jmpbuf, %struct.__jmp_buf_tag* %[[ARRAYDECAY1]], i32 1)
; CHECK-NEXT: %[[__THREW__VAL:.*]] = load i32, i32* @[[__THREW__]]
; CHECK-NEXT: store i32 0, i32* @[[__THREW__]]
; CHECK-NEXT: %[[CMP0:.*]] = icmp ne i32 %__THREW__.val, 0
; CHECK-NEXT: %[[THREWVALUE_VAL:.*]] = load i32, i32* @[[THREWVALUE]]
; CHECK-NEXT: %[[CMP1:.*]] = icmp ne i32 %[[THREWVALUE_VAL]], 0
; CHECK-NEXT: %[[CMP:.*]] = and i1 %[[CMP0]], %[[CMP1]]
; CHECK-NEXT: br i1 %[[CMP]], label %if.then1, label %if.else1

; CHECK: entry.split.split:
; CHECK-NEXT: unreachable

; CHECK: if.then1:
; CHECK-NEXT: %[[__THREW__VAL_I32P:.*]] = inttoptr i32 %[[__THREW__VAL]] to i32*
; CHECK-NEXT: %[[__THREW__VAL_I32P_LOADED:.*]] = load i32, i32* %[[__THREW__VAL_I32P]]
; CHECK-NEXT: %[[LABEL:.*]] = call i32 @testSetjmp(i32 %[[__THREW__VAL_I32P_LOADED]], i32* %[[SETJMP_TABLE1]], i32 %[[SETJMP_TABLE_SIZE1]])
; CHECK-NEXT: %[[CMP:.*]] = icmp eq i32 %[[LABEL]], 0
; CHECK-NEXT: br i1 %[[CMP]], label %if.then2, label %if.end2

; CHECK: if.else1:
; CHECK-NEXT: br label %if.end

; CHECK: if.end:
; CHECK-NEXT: %[[LABEL_PHI:.*]] = phi i32 [ %[[LABEL:.*]], %if.end2 ], [ -1, %if.else1 ]
; CHECK-NEXT: %[[LONGJMP_RESULT]] = load i32, i32* @[[TEMPRET0]]
; CHECK-NEXT: switch i32 %[[LABEL_PHI]], label %entry.split.split [
; CHECK-NEXT:   i32 1, label %entry.split
; CHECK-NEXT: ]

; CHECK: if.then2:
; CHECK-NEXT: call void @emscripten_longjmp(i32 %[[__THREW__VAL]], i32 %[[THREWVALUE_VAL]])
; CHECK-NEXT: unreachable

; CHECK: if.end2:
; CHECK-NEXT: store i32 %[[THREWVALUE_VAL]], i32* @[[TEMPRET0]]
; CHECK-NEXT: br label %if.end
}

; Test a case of a function call (which is not longjmp) after a setjmp
define hidden void @setjmp_other() {
; CHECK-LABEL: @setjmp_other
entry:
  %buf = alloca [1 x %struct.__jmp_buf_tag], align 16
  %arraydecay = getelementptr inbounds [1 x %struct.__jmp_buf_tag], [1 x %struct.__jmp_buf_tag]* %buf, i32 0, i32 0
  %call = call i32 @setjmp(%struct.__jmp_buf_tag* %arraydecay) #0
  call void @foo()
  ret void
; CHECK: entry:
; CHECK: %[[SETJMP_TABLE:.*]] = call i32* @saveSetjmp(

; CHECK: entry.split:
; CHECK: call void @__invoke_void(void ()* @foo)

; CHECK: entry.split.split:
; CHECK-NEXT: %[[BUF:.*]] = bitcast i32* %[[SETJMP_TABLE]] to i8*
; CHECK-NEXT: tail call void @free(i8* %[[BUF]])
; CHECK-NEXT: ret void
}

; Test a case when a function call is within try-catch, after a setjmp
define hidden void @exception_and_longjmp() #3 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
; CHECK-LABEL: @exception_and_longjmp
entry:
  %buf = alloca [1 x %struct.__jmp_buf_tag], align 16
  %arraydecay = getelementptr inbounds [1 x %struct.__jmp_buf_tag], [1 x %struct.__jmp_buf_tag]* %buf, i32 0, i32 0
  %call = call i32 @setjmp(%struct.__jmp_buf_tag* %arraydecay) #0
  invoke void @foo()
          to label %try.cont unwind label %lpad

; CHECK: entry.split:
; CHECK: store i32 0, i32* @[[__THREW__]]
; CHECK-NEXT: call void @__invoke_void(void ()* @foo)
; CHECK-NEXT: %[[__THREW__VAL:.*]] = load i32, i32* @[[__THREW__]]
; CHECK-NEXT: store i32 0, i32* @[[__THREW__]]
; CHECK-NEXT: %[[CMP0:.*]] = icmp ne i32 %[[__THREW__VAL]], 0
; CHECK-NEXT: %[[THREWVALUE_VAL:.*]] = load i32, i32* @[[THREWVALUE]]
; CHECK-NEXT: %[[CMP1:.*]] = icmp ne i32 %[[THREWVALUE_VAL]], 0
; CHECK-NEXT: %[[CMP:.*]] = and i1 %[[CMP0]], %[[CMP1]]
; CHECK-NEXT: br i1 %[[CMP]], label %if.then1, label %if.else1

; CHECK: entry.split.split:
; CHECK-NEXT: %[[CMP:.*]] = icmp eq i32 %[[__THREW__VAL]], 1
; CHECK-NEXT: br i1 %[[CMP]], label %lpad, label %try.cont

lpad:                                             ; preds = %entry
  %0 = landingpad { i8*, i32 }
          catch i8* null
  %1 = extractvalue { i8*, i32 } %0, 0
  %2 = extractvalue { i8*, i32 } %0, 1
  %3 = call i8* @__cxa_begin_catch(i8* %1) #2
  call void @__cxa_end_catch()
  br label %try.cont

try.cont:                                         ; preds = %entry, %lpad
  ret void
}

; Test SSA validity
define hidden void @ssa(i32 %n) {
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
; CHECK-NEXT: %[[SETJMP_TABLE_SIZE1:.*]] = load i32, i32* @[[TEMPRET0]]

; CHECK: if.then.split:
; CHECK: %[[VAR1:.*]] = phi i32 [ %[[VAR0]], %if.then ], [ %[[VAR2:.*]], %if.end3 ]
; CHECK: %[[SETJMP_TABLE_SIZE2:.*]] = phi i32 [ %[[SETJMP_TABLE_SIZE1]], %if.then ], [ %[[SETJMP_TABLE_SIZE3:.*]], %if.end3 ]
; CHECK: %[[SETJMP_TABLE2:.*]] = phi i32* [ %[[SETJMP_TABLE1]], %if.then ], [ %[[SETJMP_TABLE3:.*]], %if.end3 ]
; CHECK: store i32 %[[VAR1]], i32* @global_var, align 4

if.end:                                           ; preds = %if.then, %entry
  %arraydecay1 = getelementptr inbounds [1 x %struct.__jmp_buf_tag], [1 x %struct.__jmp_buf_tag]* %buf, i32 0, i32 0
  call void @longjmp(%struct.__jmp_buf_tag* %arraydecay1, i32 5) #1
  unreachable
; CHECK: if.end:
; CHECK: %[[VAR2]] = phi i32 [ %[[VAR1]], %if.then.split ], [ undef, %entry ]
; CHECK: %[[SETJMP_TABLE_SIZE3]] = phi i32 [ %[[SETJMP_TABLE_SIZE2]], %if.then.split ], [ %[[SETJMP_TABLE_SIZE0]], %entry ]
; CHECK: %[[SETJMP_TABLE3]] = phi i32* [ %[[SETJMP_TABLE2]], %if.then.split ], [ %[[SETJMP_TABLE0]], %entry ]
}

; Test a case when a function only calls other functions that are neither setjmp nor longjmp
define hidden void @only_other_func() {
entry:
  call void @foo()
  ret void
; CHECK: call void @foo()
}

; Test a case when a function only calls longjmp and not setjmp
define hidden void @only_longjmp() {
entry:
  %buf = alloca [1 x %struct.__jmp_buf_tag], align 16
  %arraydecay = getelementptr inbounds [1 x %struct.__jmp_buf_tag], [1 x %struct.__jmp_buf_tag]* %buf, i32 0, i32 0
  call void @longjmp(%struct.__jmp_buf_tag* %arraydecay, i32 5) #1
  unreachable
; CHECK: %[[ARRAYDECAY:.*]] = getelementptr inbounds
; CHECK-NEXT: call void @emscripten_longjmp_jmpbuf(%struct.__jmp_buf_tag* %[[ARRAYDECAY]], i32 5) #1
}

declare void @foo()
; Function Attrs: returns_twice
declare i32 @setjmp(%struct.__jmp_buf_tag*) #0
; Function Attrs: noreturn
declare void @longjmp(%struct.__jmp_buf_tag*, i32) #1
declare i32 @__gxx_personality_v0(...)
declare i8* @__cxa_begin_catch(i8*)
declare void @__cxa_end_catch()
declare i8* @malloc(i32)
declare void @free(i8*)

; JS glue functions and invoke wrappers declaration
; CHECK-DAG: declare i32* @saveSetjmp(%struct.__jmp_buf_tag*, i32, i32*, i32)
; CHECK-DAG: declare i32 @testSetjmp(i32, i32*, i32)
; CHECK-DAG: declare void @emscripten_longjmp_jmpbuf(%struct.__jmp_buf_tag*, i32)
; CHECK-DAG: declare void @emscripten_longjmp(i32, i32)
; CHECK-DAG: declare void @__invoke_void(void ()*)
; CHECK-DAG: declare void @"__invoke_void_%struct.__jmp_buf_tag*_i32"(void (%struct.__jmp_buf_tag*, i32)*, %struct.__jmp_buf_tag*, i32)

attributes #0 = { returns_twice }
attributes #1 = { noreturn }
attributes #2 = { nounwind }
