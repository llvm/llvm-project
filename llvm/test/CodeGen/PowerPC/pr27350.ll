; RUN: llc -verify-machineinstrs -mcpu=ppc64le -mtriple=powerpc64le-unknown-linux-gnu < %s

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i1) #0

; Function Attrs: nounwind
define internal fastcc void @foo() unnamed_addr #1 align 2 {
entry:
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 undef, i8* align 8 null, i64 16, i1 false)
  %0 = load <2 x i64>, <2 x i64>* null, align 8
  %1 = extractelement <2 x i64> %0, i32 1
  %.fca.1.insert159.i = insertvalue [2 x i64] undef, i64 %1, 1
  tail call fastcc void @bar([2 x i64] undef, [2 x i64] %.fca.1.insert159.i) #2
  unreachable
}

; Function Attrs: nounwind
declare fastcc void @bar([2 x i64], [2 x i64]) unnamed_addr #1 align 2

attributes #0 = { argmemonly nounwind }
attributes #1 = { nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="ppc64le" "target-features"="+altivec,+bpermd,+crypto,+direct-move,+extdiv,+power8-vector,+vsx,-qpx" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.9.0 (trunk) (llvm/trunk 266222)"}
