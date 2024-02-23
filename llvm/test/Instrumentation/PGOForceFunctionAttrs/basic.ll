; RUN: opt < %s -passes=pgo-force-function-attrs -pgo-kind=pgo-instr-use-pipeline -S -pgo-cold-func-opt=default | FileCheck %s --check-prefixes=NONE,CHECK
; RUN: opt < %s -passes=pgo-force-function-attrs -pgo-kind=pgo-instr-use-pipeline -S -pgo-cold-func-opt=optsize | FileCheck %s --check-prefixes=OPTSIZE,CHECK
; RUN: opt < %s -passes=pgo-force-function-attrs -pgo-kind=pgo-instr-use-pipeline -S -pgo-cold-func-opt=minsize | FileCheck %s --check-prefixes=MINSIZE,CHECK
; RUN: opt < %s -passes=pgo-force-function-attrs -pgo-kind=pgo-instr-use-pipeline -S -pgo-cold-func-opt=optnone | FileCheck %s --check-prefixes=OPTNONE,CHECK

; Should be no changes without profile data
; RUN: opt < %s -passes=pgo-force-function-attrs                                  -S -pgo-cold-func-opt=minsize | FileCheck %s --check-prefixes=NONE,CHECK

; NONE-NOT: Function Attrs:
; OPTSIZE: Function Attrs: optsize{{$}}
; MINSIZE: Function Attrs: minsize{{$}}
; OPTNONE: Function Attrs: noinline optnone{{$}}
; CHECK: define void @cold()

; CHECK: Function Attrs: optsize{{$}}
; CHECK-NEXT: define void @cold_optsize()

; CHECK: Function Attrs: minsize{{$}}
; CHECK-NEXT: define void @cold_minsize()

; CHECK: Function Attrs: noinline optnone{{$}}
; CHECK-NEXT: define void @cold_optnone()

; NONE: Function Attrs: cold{{$}}
; OPTSIZE: Function Attrs: cold optsize{{$}}
; MINSIZE: Function Attrs: cold minsize{{$}}
; OPTNONE: Function Attrs: cold noinline optnone{{$}}
; CHECK-NEXT: define void @cold_attr()

; CHECK-NOT: Function Attrs: {{.*}}optsize
; CHECK-NOT: Function Attrs: {{.*}}minsize
; CHECK-NOT: Function Attrs: {{.*}}optnone

@s = global i32 0

define void @cold() !prof !27 {
  store i32 1, ptr @s, align 4
  ret void
}

define void @cold_optsize() optsize !prof !27 {
  store i32 1, ptr @s, align 4
  ret void
}

define void @cold_minsize() minsize !prof !27 {
  store i32 1, ptr @s, align 4
  ret void
}

define void @cold_optnone() noinline optnone !prof !27 {
  store i32 1, ptr @s, align 4
  ret void
}

define void @cold_attr() cold {
  store i32 1, ptr @s, align 4
  ret void
}

define void @hot() !prof !28 {
  %l = load i32, ptr @s, align 4
  %add = add nsw i32 %l, 4
  store i32 %add, ptr @s, align 4
  ret void
}

attributes #0 = { optsize }
attributes #1 = { minsize }
attributes #2 = { noinline optnone }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"ProfileSummary", !1}
!1 = !{!2, !3, !4, !5, !6, !7, !8, !9}
!2 = !{!"ProfileFormat", !"InstrProf"}
!3 = !{!"TotalCount", i64 9040}
!4 = !{!"MaxCount", i64 9000}
!5 = !{!"MaxInternalCount", i64 0}
!6 = !{!"MaxFunctionCount", i64 9000}
!7 = !{!"NumCounts", i64 5}
!8 = !{!"NumFunctions", i64 5}
!9 = !{!"DetailedSummary", !10}
!10 = !{!11, !12, !13, !14, !15, !16, !17, !18, !19, !20, !21, !22, !23, !24, !25, !26}
!11 = !{i32 10000, i64 9000, i32 1}
!12 = !{i32 100000, i64 9000, i32 1}
!13 = !{i32 200000, i64 9000, i32 1}
!14 = !{i32 300000, i64 9000, i32 1}
!15 = !{i32 400000, i64 9000, i32 1}
!16 = !{i32 500000, i64 9000, i32 1}
!17 = !{i32 600000, i64 9000, i32 1}
!18 = !{i32 700000, i64 9000, i32 1}
!19 = !{i32 800000, i64 9000, i32 1}
!20 = !{i32 900000, i64 9000, i32 1}
!21 = !{i32 950000, i64 9000, i32 1}
!22 = !{i32 990000, i64 9000, i32 1}
!23 = !{i32 999000, i64 10, i32 5}
!24 = !{i32 999900, i64 10, i32 5}
!25 = !{i32 999990, i64 10, i32 5}
!26 = !{i32 999999, i64 10, i32 5}
!27 = !{!"function_entry_count", i64 10}
!28 = !{!"function_entry_count", i64 9000}
