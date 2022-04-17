; RUN: llc < %s -mtriple=nvptx-unknown-unknown | FileCheck %s

%struct.St8x4 = type { [4 x i64] }

define dso_local void @caller_St8x4(ptr nocapture noundef readonly byval(%struct.St8x4) align 8 %in, ptr nocapture noundef writeonly %ret) {
  %call = tail call fastcc [4 x i64] @callee_St8x4(ptr noundef nonnull byval(%struct.St8x4) align 8 %in) #2
  %.fca.0.extract = extractvalue [4 x i64] %call, 0
  %.fca.1.extract = extractvalue [4 x i64] %call, 1
  %.fca.2.extract = extractvalue [4 x i64] %call, 2
  %.fca.3.extract = extractvalue [4 x i64] %call, 3
  store i64 %.fca.0.extract, ptr %ret, align 8
  %ref.tmp.sroa.4.0..sroa_idx = getelementptr inbounds i8, ptr %ret, i64 8
  store i64 %.fca.1.extract, ptr %ref.tmp.sroa.4.0..sroa_idx, align 8
  %ref.tmp.sroa.5.0..sroa_idx = getelementptr inbounds i8, ptr %ret, i64 16
  store i64 %.fca.2.extract, ptr %ref.tmp.sroa.5.0..sroa_idx, align 8
  %ref.tmp.sroa.6.0..sroa_idx = getelementptr inbounds i8, ptr %ret, i64 24
  store i64 %.fca.3.extract, ptr %ref.tmp.sroa.6.0..sroa_idx, align 8
  ret void
}

define internal fastcc [4 x i64] @callee_St8x4(ptr nocapture noundef readonly byval(%struct.St8x4) align 8 %in) {
  %1 = load i64, ptr %in, align 8
  %arrayidx.1 = getelementptr inbounds [4 x i64], ptr %in, i64 0, i64 1
  %2 = load i64, ptr %arrayidx.1, align 8
  %arrayidx.2 = getelementptr inbounds [4 x i64], ptr %in, i64 0, i64 2
  %3 = load i64, ptr %arrayidx.2, align 8
  %arrayidx.3 = getelementptr inbounds [4 x i64], ptr %in, i64 0, i64 3
  %4 = load i64, ptr %arrayidx.3, align 8
  %5 = insertvalue [4 x i64] poison, i64 %1, 0
  %6 = insertvalue [4 x i64] %5, i64 %2, 1
  %7 = insertvalue [4 x i64] %6, i64 %3, 2
  %oldret = insertvalue [4 x i64] %7, i64 %4, 3
  ret [4 x i64] %oldret
}

define void @call_void() {
  ret void
}
