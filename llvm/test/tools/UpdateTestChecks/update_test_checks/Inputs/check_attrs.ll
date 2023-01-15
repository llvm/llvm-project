; RUN: opt -aa-pipeline=basic-aa -passes=attributor -S < %s | FileCheck %s --check-prefixes=CHECK,IS__TUNIT____
; RUN: opt -aa-pipeline=basic-aa -passes=attributor-cgscc -S < %s | FileCheck %s --check-prefixes=CHECK,IS__CGSCC____

%struct.RT = type { i8, [10 x [20 x i32]], i8 }
%struct.ST = type { i32, double, %struct.RT }

define ptr @foo(ptr %s) nounwind uwtable readnone optsize ssp {
entry:
  %arrayidx = getelementptr inbounds %struct.ST, ptr %s, i64 1, i32 2, i32 1, i64 5, i64 13
  ret ptr %arrayidx
}
