; RUN: opt -S -passes='require<profile-summary>,function(codegenprepare)' -mtriple=x86_64-linux < %s | FileCheck %s
;
; Ensure that blocks that only contain @llvm.assume are removed completely
; during CodeGenPrepare.

; CHECK-LABEL: @simple(
; CHECK-NEXT: end:
; CHECK-NEXT: ret void
define void @simple(i64 %addr, i1 %assumption) {
  %cmp1 = icmp eq i64 %addr, 0
  br i1 %cmp1, label %do_assume, label %end

do_assume:
  tail call void @llvm.assume(i1 %assumption)
  br label %end

end:
  ret void
}

; CHECK-LABEL: @complex_assume(
; CHECK-NEXT: end:
; CHECK-NEXT: ret void
define void @complex_assume(i64 %addr, i1 %assumption_a, i1 %assumption_b,
                          i64 %val_a, i64 %val_b) {
  %cmp1 = icmp eq i64 %addr, 0
  br i1 %cmp1, label %do_assume, label %end

do_assume:
  call void @llvm.assume(i1 %assumption_a)
  call void @llvm.assume(i1 %assumption_b)
  %val_xor = xor i64 %val_a, %val_b
  %val_shifted = lshr i64 %val_xor, 7
  %assumption_c = trunc i64 %val_shifted to i1
  call void @llvm.assume(i1 %assumption_c)
  %assumption_d = call i1 @readonly_func(i64 %val_b)
  call void @llvm.assume(i1 %assumption_d)
  br label %end

end:
  ret void
}

declare void @llvm.assume(i1 noundef)
declare i1 @readonly_func(i64) nounwind readonly willreturn;

