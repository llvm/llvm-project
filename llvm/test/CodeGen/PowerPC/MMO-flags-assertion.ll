; RUN: llc < %s -mtriple powerpc64le-unknown-linux-gnu

; void llvm::MachineMemOperand::refineAlignment(const llvm::MachineMemOperand*):
; Assertion `MMO->getFlags() == getFlags() && "Flags mismatch !"' failed.

declare void @_Z3fn11F(ptr byval(%class.F) align 8) local_unnamed_addr
declare void @llvm.memcpy.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i1)
declare signext i32 @_ZN1F11isGlobalRegEv(ptr) local_unnamed_addr
declare void @llvm.lifetime.start.p0(i64, ptr nocapture)
declare void @_Z10EmitLValuev(ptr sret(%class.F)) local_unnamed_addr
declare void @llvm.lifetime.end.p0(i64, ptr nocapture)

%class.F = type { i32, i64, i8, [64 x i8], i8, ptr }

define signext i32 @_Z29EmitOMPAtomicSimpleUpdateExpr1F(ptr byval(%class.F) align 8 %p1) local_unnamed_addr {
entry:
  call void @_Z3fn11F(ptr byval(%class.F) nonnull align 8 %p1)
  %call = call signext i32 @_ZN1F11isGlobalRegEv(ptr nonnull %p1)
  ret i32 %call
}

define void @_Z3fn2v() local_unnamed_addr {
entry:
  %agg.tmp1 = alloca %class.F, align 8
  %XLValue = alloca %class.F, align 8
  call void @llvm.lifetime.start.p0(i64 96, ptr nonnull %XLValue)
  call void @_Z10EmitLValuev(ptr nonnull sret(%class.F) %XLValue)
  call void @llvm.lifetime.start.p0(i64 96, ptr nonnull %agg.tmp1)
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 nonnull %agg.tmp1, ptr align 8 nonnull %XLValue, i64 96, i1 false)
  call void @_Z3fn11F(ptr byval(%class.F) nonnull align 8 %XLValue)
  %call.i = call signext i32 @_ZN1F11isGlobalRegEv(ptr nonnull %agg.tmp1)
  call void @llvm.lifetime.end.p0(i64 96, ptr nonnull %agg.tmp1)
  call void @llvm.lifetime.end.p0(i64 96, ptr nonnull %XLValue)
  ret void
}
