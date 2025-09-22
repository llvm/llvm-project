; RUN: opt < %s -passes=pgo-icall-prom -icp-allow-decls=false -icp-allow-hot-only=true -icp-allow-candidate-skip=false -S -pass-remarks=pgo-icall-prom 2>&1 | FileCheck %s --check-prefix=REMARK1
; RUN: opt < %s -passes=pgo-icall-prom -icp-allow-decls=true -icp-allow-hot-only=true -icp-allow-candidate-skip=false -S -pass-remarks=pgo-icall-prom 2>&1 | FileCheck %s --check-prefixes=REMARK1,REMARK2
; RUN: opt < %s -passes=pgo-icall-prom -icp-allow-decls=false -icp-allow-hot-only=true -icp-allow-candidate-skip=false -S -pass-remarks=pgo-icall-prom 2>&1 | FileCheck %s --check-prefix=REMARK1
; RUN: opt < %s -passes=pgo-icall-prom -icp-allow-decls=false -icp-allow-hot-only=false -icp-allow-candidate-skip=false -S -pass-remarks=pgo-icall-prom 2>&1 | FileCheck %s --check-prefixes=REMARK1
; RUN: opt < %s -passes=pgo-icall-prom -icp-allow-decls=false -icp-allow-hot-only=false -icp-allow-candidate-skip=true -S -pass-remarks=pgo-icall-prom 2>&1 | FileCheck %s --check-prefixes=REMARK1,REMARK3
; RUN: opt < %s -passes=pgo-icall-prom -icp-allow-decls=true -icp-allow-hot-only=false -icp-allow-candidate-skip=true -S -pass-remarks=pgo-icall-prom 2>&1 | FileCheck %s --check-prefixes=REMARK1,REMARK2,REMARK4,REMARK5
; RUN: opt < %s -passes=pgo-icall-prom -icp-allow-decls=false -icp-allow-hot-only=false -icp-allow-candidate-skip=true -S -pass-remarks=pgo-icall-prom 2>&1 | FileCheck %s --check-prefixes=REMARK6,REMARK1,REMARK3
; RUN: opt < %s -passes=pgo-icall-prom -icp-allow-decls=false -icp-allow-hot-only=false -icp-allow-candidate-skip=true -S | FileCheck %s --check-prefix=METADATA

; REMARK6: remark: <unknown>:0:0: Promote indirect call to add with count 20000 out of 60000
; REMARK2: remark: <unknown>:0:0: Promote indirect call to sub with count 40000 out of 60000
; REMARK2: remark: <unknown>:0:0: Promote indirect call to add with count 20000 out of 20000
; REMARK1: remark: <unknown>:0:0: Promote indirect call to add with count 10000 out of 10000
; REMARK3: remark: <unknown>:0:0: Promote indirect call to add with count 200 out of 400
; REMARK4: remark: <unknown>:0:0: Promote indirect call to sub with count 200 out of 400
; REMARK5: remark: <unknown>:0:0: Promote indirect call to add with count 200 out of 200

@math = dso_local local_unnamed_addr global ptr null, align 8

define dso_local i32 @add(i32 noundef %a, i32 noundef %b) !prof !34 {
entry:
  %add = add nsw i32 %a, %b
  ret i32 %add
}

define dso_local range(i32 0, 2) i32 @main() !prof !35 {
entry:
  call void @setup(i32 noundef 0)
  br label %for.cond

for.cond:
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %sum.0 = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %cmp = icmp samesign ult i32 %i.0, 50000
  br i1 %cmp, label %for.body, label %for.end, !prof !36

for.body:
  %0 = load ptr, ptr @math, align 8, !tbaa !37
  %call = call i32 %0(i32 noundef %i.0, i32 noundef %i.0), !prof !41
; METADATA: %call = call i32 %0(i32 noundef %i.0, i32 noundef %i.0), !prof ![[NEWVP:[0-9]+]]
; METADATA: ![[NEWVP]] = !{!"VP", i32 0, i64 40000, i64 -455885480058394486, i64 40000}
  %add = add nsw i32 %sum.0, %call
  %inc = add nuw nsw i32 %i.0, 1
  br label %for.cond, !llvm.loop !42

for.end:
  call void @setup(i32 noundef 1)
  br label %for.cond1

for.cond1:
  %i.1 = phi i32 [ 0, %for.end ], [ %inc7, %for.body3 ]
  %sum.1 = phi i32 [ %sum.0, %for.end ], [ %add5, %for.body3 ]
  %cmp2 = icmp samesign ult i32 %i.1, 10000
  br i1 %cmp2, label %for.body3, label %for.cond9, !prof !44

for.body3:
  %1 = load ptr, ptr @math, align 8, !tbaa !37
  %call4 = call i32 %1(i32 noundef %i.1, i32 noundef %i.1), !prof !45
  %add5 = add nsw i32 %sum.1, %call4
  %inc7 = add nuw nsw i32 %i.1, 1
  br label %for.cond1, !llvm.loop !46

for.cond9:
  %i.2 = phi i32 [ %inc15, %for.body11 ], [ 0, %for.cond1 ]
  %sum.2 = phi i32 [ %add13, %for.body11 ], [ %sum.1, %for.cond1 ]
  %cmp10 = icmp samesign ult i32 %i.2, 400
  br i1 %cmp10, label %for.body11, label %for.cond17, !prof !47

for.body11:
  call void @setup(i32 noundef %i.2)
  %2 = load ptr, ptr @math, align 8, !tbaa !37
  %call12 = call i32 %2(i32 noundef %i.2, i32 noundef %i.2), !prof !48
  %add13 = add nsw i32 %sum.2, %call12
  %inc15 = add nuw nsw i32 %i.2, 1
  br label %for.cond9, !llvm.loop !49

for.cond17:
  %i.3 = phi i32 [ %inc25, %for.body19 ], [ 0, %for.cond9 ]
  %sum.3 = phi i32 [ %add23, %for.body19 ], [ %sum.2, %for.cond9 ]
  %cmp18 = icmp samesign ult i32 %i.3, 400
  br i1 %cmp18, label %for.body19, label %for.end26, !prof !47

for.body19:
  %add.i = shl nuw nsw i32 %i.3, 1
  %add21 = add nsw i32 %sum.3, %add.i
  %call22 = call i32 @sub(i32 noundef %i.3, i32 noundef %i.3)
  %add23 = add nsw i32 %add21, %call22
  %inc25 = add nuw nsw i32 %i.3, 1
  br label %for.cond17, !llvm.loop !50

for.end26:
  %cmp27 = icmp slt i32 %sum.3, 11
  %. = zext i1 %cmp27 to i32
  ret i32 %.
}

declare void @setup(i32 noundef)

declare i32 @sub(i32 noundef, i32 noundef)

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!33}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 1, !"ProfileSummary", !5}
!5 = !{!6, !7, !8, !9, !10, !11, !12, !13, !14, !15}
!6 = !{!"ProfileFormat", !"InstrProf"}
!7 = !{!"TotalCount", i64 122204}
!8 = !{!"MaxCount", i64 50600}
!9 = !{!"MaxInternalCount", i64 10000}
!10 = !{!"MaxFunctionCount", i64 50600}
!11 = !{!"NumCounts", i64 9}
!12 = !{!"NumFunctions", i64 4}
!13 = !{!"IsPartialProfile", i64 0}
!14 = !{!"PartialProfileRatio", double 0.000000e+00}
!15 = !{!"DetailedSummary", !16}
!16 = !{!17, !18, !19, !20, !21, !22, !23, !24, !25, !26, !27, !28, !29, !30, !31, !32}
!17 = !{i32 10000, i64 50600, i32 1}
!18 = !{i32 100000, i64 50600, i32 1}
!19 = !{i32 200000, i64 50600, i32 1}
!20 = !{i32 300000, i64 50600, i32 1}
!21 = !{i32 400000, i64 50600, i32 1}
!22 = !{i32 500000, i64 50000, i32 2}
!23 = !{i32 600000, i64 50000, i32 2}
!24 = !{i32 700000, i64 50000, i32 2}
!25 = !{i32 800000, i64 50000, i32 2}
!26 = !{i32 900000, i64 10200, i32 3}
!27 = !{i32 950000, i64 10000, i32 4}
!28 = !{i32 990000, i64 402, i32 5}
!29 = !{i32 999000, i64 201, i32 8}
!30 = !{i32 999900, i64 201, i32 8}
!31 = !{i32 999990, i64 201, i32 8}
!32 = !{i32 999999, i64 201, i32 8}
!33 = !{!"clang version 22.0.0git (git@github.com:llvm/llvm-project.git ac20b28c2be26061e63dceac0915f97ece2273ac)"}
!34 = !{!"function_entry_count", i64 10200}
!35 = !{!"function_entry_count", i64 1}
!36 = !{!"branch_weights", i32 50000, i32 1}
!37 = !{!38, !38, i64 0}
!38 = !{!"any pointer", !39, i64 0}
!39 = !{!"omnipotent char", !40, i64 0}
!40 = !{!"Simple C/C++ TBAA"}
!41 = !{!"VP", i32 0, i64 60000, i64 -455885480058394486, i64 40000, i64 2232412992676883508, i64 20000}
!42 = distinct !{!42, !43}
!43 = !{!"llvm.loop.mustprogress"}
!44 = !{!"branch_weights", i32 10000, i32 1}
!45 = !{!"VP", i32 0, i64 10000, i64 2232412992676883508, i64 10000}
!46 = distinct !{!46, !43}
!47 = !{!"branch_weights", i32 400, i32 1}
!48 = !{!"VP", i32 0, i64 400, i64 -455885480058394486, i64 200, i64 2232412992676883508, i64 200}
!49 = distinct !{!49, !43}
!50 = distinct !{!50, !43}
