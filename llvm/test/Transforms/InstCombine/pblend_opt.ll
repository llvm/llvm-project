; ModuleID = 'pblend.ll'
source_filename = "pblend.ll"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none)
define <16 x i8> @simple_blend(<16 x i8> %x, <16 x i8> %y, <4 x i32> %a) local_unnamed_addr #0 {
entry:
  %cmp = icmp sgt <4 x i32> %a, zeroinitializer
  %sext = sext <4 x i1> %cmp to <4 x i32>
  %mask = bitcast <4 x i32> %sext to <16 x i8>
  %res = tail call <16 x i8> @llvm.x86.sse41.pblendvb(<16 x i8> %x, <16 x i8> %y, <16 x i8> %mask)
  ret <16 x i8> %res
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <16 x i8> @llvm.x86.sse41.pblendvb(<16 x i8>, <16 x i8>, <16 x i8>) #1

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(none) }
