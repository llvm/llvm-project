; Test that instrumentaiton works fine for the case of failing the split critical edges.
; RUN: opt < %s -passes=pgo-instr-gen -S | FileCheck %s --check-prefix=GEN
; RUN: llvm-profdata merge %S/Inputs/PR41279.proftext -o %t.profdata
; RUN: opt < %s -passes=pgo-instr-use -pgo-test-profile-file=%t.profdata -S | FileCheck %s --check-prefix=USE

declare void @f3(ptr, ptr, i64)
declare { ptr, i64 } @f0(ptr)
declare i64 @f1()
declare void @invok2(ptr, ptr noalias readonly align 1, i64)
declare void @invok1(ptr, ptr, i64)
declare i32 @__CxxFrameHandler3(...)

define void @foo(ptr, ptr) personality ptr @__CxxFrameHandler3 {
; USE-LABEL: @foo
; USE-SAME: !prof ![[FUNC_ENTRY_COUNT:[0-9]+]]

  %3 = alloca i8, align 1
  store i8 0, ptr %3, align 1
  %4 = call i64 @f1()
  %5 = icmp ult i64 %4, 32
  br i1 %5, label %7, label %13
; USE: br i1 %5, label %7, label %13
; USE-SAME: !prof ![[BW_ENTRY1:[0-9]+]]

6:
  cleanupret from %17 unwind to caller
; GEN: 6:
; GEN:  call void @llvm.instrprof.increment(ptr @__profn_foo, i64 {{[0-9]+}}, i32 4, i32 2)

7:
  store i8 1, ptr %3, align 1
  %8 = call { ptr, i64 } @f0(ptr %0)
  %9 = extractvalue { ptr, i64 } %8, 0
  %10 = extractvalue { ptr, i64 } %8, 1
  invoke void @invok1(ptr %1, ptr %0, i64 1)
          to label %11 unwind label %16
; GEN: 7:
; GEN-NOT: call void @llvm.instrprof.increment

11:
  store i8 0, ptr %3, align 1
  invoke void @invok2(ptr %1, ptr noalias readonly align 1 %9, i64 %10)
          to label %12 unwind label %16
; GEN: 11:
; GEN-NOT: call void @llvm.instrprof.increment

12:
  store i8 0, ptr %3, align 1
  br label %14
; GEN: 12:
; GEN:  call void @llvm.instrprof.increment(ptr @__profn_foo, i64 {{[0-9]+}}, i32 4, i32 1)

13:
  call void @f3(ptr %0, ptr %1, i64 1)
  br label %14
; GEN: 13:
; GEN:  call void @llvm.instrprof.increment(ptr @__profn_foo, i64 {{[0-9]+}}, i32 4, i32 0)

14:
  ret void

15:
  store i8 0, ptr %3, align 1
  br label %6
; GEN: 15:
; GEN:  call void @llvm.instrprof.increment(ptr @__profn_foo, i64 {{[0-9]+}}, i32 4, i32 3)

16:
  %17 = cleanuppad within none []
  %18 = load i8, ptr %3, align 1
  %19 = trunc i8 %18 to i1
  br i1 %19, label %15, label %6
; USE: br i1 %19, label %15, label %6
; USE-SAME: !prof ![[BW_ENTRY2:[0-9]+]]
}

; USE-DAG: {{![0-9]+}} = !{i32 1, !"ProfileSummary", {{![0-9]+}}}
; USE-DAG: {{![0-9]+}} = !{!"DetailedSummary", {{![0-9]+}}}
; USE-DAG: ![[FUNC_ENTRY_COUNT]] = !{!"function_entry_count", i64 8}
; USE-DAG: ![[BW_ENTRY1]] = !{!"branch_weights", i32 5, i32 3}
; USE-DAG: ![[BW_ENTRY2]] = !{!"branch_weights", i32 2, i32 1}

