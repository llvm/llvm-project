; RUN: llc %s -mtriple=s390x-linux-gnu -mcpu=zEC12 -o - -verify-machineinstrs \
; RUN:   | FileCheck %s

@PawnTT = dso_local local_unnamed_addr global [2048 x i8] zeroinitializer, align 2

define dso_local void @clear_pawn_tt() local_unnamed_addr #0 {
entry:
  call void @llvm.memset.p0i8.i64(i8* nonnull align 2 dereferenceable(2048) getelementptr inbounds ([2048 x i8], [2048 x i8]* @PawnTT, i64 0, i64 0), i8 0, i64 2048, i1 false)
  ret void
}

declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1 immarg) #1

attributes #0 = { nofree nounwind writeonly "fentry-call"="true" }
attributes #1 = { argmemonly nofree nosync nounwind willreturn writeonly }

; CHECK: clear_pawn_tt: # @clear_pawn_tt
; CHECK-NEXT: # %bb.0:
; CHECK-NEXT: brasl %r0, __fentry__@PLT
