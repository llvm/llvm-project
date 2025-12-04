; RUN: llc -march=hexagon -mv75 -mhvx -mattr=+hvx-length128b < %s
; REQUIRES: asserts

define dso_local void @vc1INTERP_PredictMB([64 x i8]* %pPredBlk) local_unnamed_addr {
entry:
  %next.gep111 = getelementptr [64 x i8], [64 x i8]* %pPredBlk, i32 0, i32 0
  %wide.load112 = load <32 x i8>, <32 x i8>* poison, align 32
  %0 = zext <32 x i8> %wide.load112 to <32 x i16>
  %1 = add nuw nsw <32 x i16> zeroinitializer, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  %2 = add nuw nsw <32 x i16> %1, %0
  %3 = lshr <32 x i16> %2, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  %4 = trunc <32 x i16> %3 to <32 x i8>
  %5 = bitcast i8* %next.gep111 to <32 x i8>*
  store <32 x i8> %4, <32 x i8>* %5, align 1
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull null)
  unreachable
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture)
