; RUN: not llc -march=bpf < %s 2> %t1
; RUN: FileCheck %s < %t1
; CHECK: error: <unknown>:0:0: in function bar { i64, i32 } (i32, i32, i32, i32, i32): aggregate returns are not supported

%struct.S = type { i32, i32, i32 }

@s = common global %struct.S zeroinitializer, align 4

; Function Attrs: nounwind readonly uwtable
define { i64, i32 } @bar(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e) #0 {
entry:
  %retval.sroa.0.0.copyload = load i64, ptr @s, align 4
  %retval.sroa.2.0.copyload = load i32, ptr getelementptr inbounds (%struct.S, ptr @s, i64 0, i32 2), align 4
  %.fca.0.insert = insertvalue { i64, i32 } undef, i64 %retval.sroa.0.0.copyload, 0
  %.fca.1.insert = insertvalue { i64, i32 } %.fca.0.insert, i32 %retval.sroa.2.0.copyload, 1
  ret { i64, i32 } %.fca.1.insert
}

; CHECK: error: <unknown>:0:0: in function baz void (ptr): aggregate returns are not supported

%struct.B = type { [100 x i64] }

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none)
define dso_local void @baz(ptr noalias nocapture sret(%struct.B) align 8 %agg.result) local_unnamed_addr #0 {
entry:
  ret void
}
