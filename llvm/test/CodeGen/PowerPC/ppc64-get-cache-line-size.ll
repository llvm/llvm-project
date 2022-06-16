; RUN: llc < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 -enable-ppc-prefetching=true | FileCheck %s
; RUN: llc < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 -enable-ppc-prefetching=true -cache-line-size=64 | FileCheck %s -check-prefix=CHECK-DCBT
; RUN: llc < %s -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr8 -enable-ppc-prefetching=true | FileCheck %s
; RUN: llc < %s -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr8 -enable-ppc-prefetching=true -cache-line-size=64 | FileCheck %s -check-prefix=CHECK-DCBT
; RUN: llc < %s -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr9 -enable-ppc-prefetching=true | FileCheck %s
; RUN: llc < %s -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr9 -enable-ppc-prefetching=true -cache-line-size=64 | FileCheck %s -check-prefix=CHECK-DCBT
; RUN: llc < %s -mtriple=ppc64-- -mcpu=a2 -enable-ppc-prefetching=true | FileCheck %s -check-prefix=CHECK-DCBT

; Function Attrs: nounwind
define signext i32 @check_cache_line() local_unnamed_addr {
entry:
  %call = tail call i32* bitcast (i32* (...)* @magici to i32* ()*)()
  %call115 = tail call signext i32 bitcast (i32 (...)* @iter to i32 ()*)()
  %cmp16 = icmp sgt i32 %call115, 0
  br i1 %cmp16, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %res.0.lcssa = phi i32 [ 0, %entry ], [ %add5, %for.body ]
  ret i32 %res.0.lcssa

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %res.017 = phi i32 [ %add5, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %call, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %0, %res.017
  %1 = add nuw nsw i64 %indvars.iv, 16
  %arrayidx4 = getelementptr inbounds i32, i32* %call, i64 %1
  %2 = load i32, i32* %arrayidx4, align 4
  %add5 = add nsw i32 %add, %2
  %indvars.iv.next = add nuw i64 %indvars.iv, 1
  %call1 = tail call signext i32 bitcast (i32 (...)* @iter to i32 ()*)()
  %3 = sext i32 %call1 to i64
  %cmp = icmp slt i64 %indvars.iv.next, %3
  br i1 %cmp, label %for.body, label %for.cond.cleanup
; CHECK-LABEL: check_cache_line
; CHECK: dcbt
; CHECK-NOT: dcbt
; CHECK: blr
; CHECK-DCBT-LABEL: check_cache_line
; CHECK-DCBT: dcbt
; CHECK-DCBT: dcbt
; CHECK-DCBT: blr
}

declare i32* @magici(...) local_unnamed_addr

declare signext i32 @iter(...) local_unnamed_addr

