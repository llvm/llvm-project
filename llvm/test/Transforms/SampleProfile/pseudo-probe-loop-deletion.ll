; RUN: opt %s -passes=loop-deletion -S | FileCheck %s --check-prefixes=CHECK

%class.Loc.95 = type { %class.Domain.96 }
%class.Domain.96 = type { %class.DomainBase.97 }
%class.DomainBase.97 = type { [3 x %struct.WrapNoInit] }
%struct.WrapNoInit = type { %class.Loc }
%class.Loc = type { %class.Domain.67 }
%class.Domain.67 = type { %class.DomainBase.68 }
%class.DomainBase.68 = type { i32 }

define dso_local void @foo(ptr %0) {
; CHECK-LABEL: @foo(
; CHECK-NEXT:    br label [[foo:%.*]]
; CHECK:       foo.exit:
; CHECK-NEXT:    ret void
;
  br label %2

2:                                                ; preds = %4, %1
  %.0.i.i = phi ptr [ undef, %1 ], [ %5, %4 ]
  %3 = icmp ne ptr %.0.i.i, %0
  br i1 %3, label %4, label %foo.exit

4:                                                ; preds = %2
  call void @llvm.pseudoprobe(i64 6878943695821059507, i64 9, i32 0, i64 -1)
  %5 = getelementptr inbounds %class.Loc.95, ptr %.0.i.i, i32 1
  br label %2

foo.exit:            ; preds = %2
  ret void
}

declare void @llvm.pseudoprobe(i64, i64, i32, i64)  #1

attributes #1 = { willreturn readnone norecurse nofree }
