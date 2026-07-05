; When optimising for size, we don't want to rewrite fputs to fwrite
; because it requires more arguments and thus extra MOVs are required.
;
; RUN: opt < %s -passes=instcombine -S | FileCheck %s
; RUN: opt < %s -passes='require<profile-summary>,function(instcombine)' -pgso -S | FileCheck %s -check-prefix=PGSO
; RUN: opt < %s -passes=instcombine -pgso=false -S | FileCheck %s -check-prefix=NPGSO

%struct._IO_FILE = type { i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i32, i16, i8, [1 x i8], ptr, i64, ptr, ptr, ptr, ptr, i32, i32, [40 x i8] }
%struct._IO_marker = type { ptr, ptr, i32 }

@.str = private unnamed_addr constant [10 x i8] c"mylog.txt\00", align 1
@.str.1 = private unnamed_addr constant [2 x i8] c"a\00", align 1
@.str.2 = private unnamed_addr constant [27 x i8] c"Hello world this is a test\00", align 1

define i32 @main() local_unnamed_addr #0 {
entry:
; CHECK-LABEL: @main(
; CHECK-NOT: call i64 @fwrite
; CHECK: call i32 @fputs

  %call = tail call ptr @fopen(ptr @.str, ptr @.str.1) #2
  %call1 = tail call i32 @fputs(ptr @.str.2, ptr %call) #2
  ret i32 0
}

declare noalias ptr @fopen(ptr nocapture readonly, ptr nocapture readonly) local_unnamed_addr #1
declare i32 @fputs(ptr nocapture readonly, ptr nocapture) local_unnamed_addr #1

attributes #0 = { nounwind optsize }
attributes #1 = { nounwind optsize  }

define i32 @main_pgso() local_unnamed_addr !prof !14 {
entry:
; PGSO-LABEL: @main_pgso(
; PGSO-NOT: call i64 @fwrite
; PGSO: call i32 @fputs
; NPGSO-LABEL: @main_pgso(
; NPGSO: call i64 @fwrite
; NPGSO-NOT: call i32 @fputs

  %call = tail call ptr @fopen(ptr @.str, ptr @.str.1) #2
  %call1 = tail call i32 @fputs(ptr @.str.2, ptr %call) #2
  ret i32 0
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"ProfileSummary", !1}
!1 = !{!2, !3, !4, !5, !6, !7, !8, !9}
!2 = !{!"ProfileFormat", !"InstrProf"}
!3 = !{!"TotalCount", i64 10000}
!4 = !{!"MaxCount", i64 10}
!5 = !{!"MaxInternalCount", i64 1}
!6 = !{!"MaxFunctionCount", i64 1000}
!7 = !{!"NumCounts", i64 3}
!8 = !{!"NumFunctions", i64 3}
!9 = !{!"DetailedSummary", !10}
!10 = !{!11, !12, !13}
!11 = !{i32 10000, i64 100, i32 1}
!12 = !{i32 999000, i64 100, i32 1}
!13 = !{i32 999999, i64 1, i32 2}
!14 = !{!"function_entry_count", i64 0}
