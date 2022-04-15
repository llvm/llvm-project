; Test that instrumentaiton works fine for the case of catchswitch stmts.
; RUN: opt < %s -passes=pgo-instr-gen -S | FileCheck %s --check-prefix=GEN
; RUN: llvm-profdata merge %S/Inputs/PR41279_2.proftext -o %t.profdata
; RUN: opt < %s -passes=pgo-instr-use -pgo-test-profile-file=%t.profdata -S | FileCheck %s --check-prefix=USE


define dso_local void @f() personality i8* bitcast (i32 (...)* @__C_specific_handler to i8*) {
; USE-LABEL: @f
; USE-SAME: !prof ![[FUNC_ENTRY_COUNT:[0-9]+]]
; USE-DAG: {{![0-9]+}} = !{i32 1, !"ProfileSummary", {{![0-9]+}}}
; USE-DAG: {{![0-9]+}} = !{!"DetailedSummary", {{![0-9]+}}}
; USE-DAG: ![[FUNC_ENTRY_COUNT]] = !{!"function_entry_count", i64 5}
entry:
  %__exception_code = alloca i32, align 4
  %__exception_code2 = alloca i32, align 4
  invoke void @f() #2
          to label %invoke.cont unwind label %catch.dispatch

catch.dispatch:
  %0 = catchswitch within none [label %__except] unwind to caller

__except:
  %1 = catchpad within %0 [i8* null]
  catchret from %1 to label %__except1

__except1:
  %2 = call i32 @llvm.eh.exceptioncode(token %1)
  store i32 %2, i32* %__exception_code, align 4
  br label %__try.cont7
;GEN:  _except1:
;GEN:    call void @llvm.instrprof.increment(i8* getelementptr inbounds ([1 x i8], [1 x i8]* @__profn_f, i32 0, i32 0), i64 {{[0-9]+}}, i32 2, i32 1)

invoke.cont:
  br label %__try.cont

__try.cont:
  invoke void @f()
          to label %invoke.cont3 unwind label %catch.dispatch4

catch.dispatch4:
  %3 = catchswitch within none [label %__except5] unwind to caller
; GEN: catch.dispatch4:
; GEN-NOT: call void @llvm.instrprof.increment

__except5:
  %4 = catchpad within %3 [i8* null]
  catchret from %4 to label %__except6

__except6:
  %5 = call i32 @llvm.eh.exceptioncode(token %4)
  store i32 %5, i32* %__exception_code2, align 4
  br label %__try.cont7

__try.cont7:
  ret void

invoke.cont3:
  br label %__try.cont7
;GEN: invoke.cont3:
;GEN:  call void @llvm.instrprof.increment(i8* getelementptr inbounds ([1 x i8], [1 x i8]* @__profn_f, i32 0, i32 0), i64 {{[0-9]+}}, i32 2, i32 0)

}

declare dso_local i32 @__C_specific_handler(...)

declare i32 @llvm.eh.exceptioncode(token)
