; RUN: llc < %s -mtriple powerpc64le-unknown-linux-gnu

; void llvm::MachineMemOperand::refineAlignment(const llvm::MachineMemOperand*):
; Assertion `MMO->getFlags() == getFlags() && "Flags mismatch !"' failed.

declare void @_Z3fn11F(%class.F* byval(%class.F) align 8) local_unnamed_addr
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1)
declare signext i32 @_ZN1F11isGlobalRegEv(%class.F*) local_unnamed_addr
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture)
declare void @_Z10EmitLValuev(%class.F* sret(%class.F)) local_unnamed_addr
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture)

%class.F = type { i32, i64, i8, [64 x i8], i8, i32* }

define signext i32 @_Z29EmitOMPAtomicSimpleUpdateExpr1F(%class.F* byval(%class.F) align 8 %p1) local_unnamed_addr {
entry:
  call void @_Z3fn11F(%class.F* byval(%class.F) nonnull align 8 %p1)
  %call = call signext i32 @_ZN1F11isGlobalRegEv(%class.F* nonnull %p1)
  ret i32 %call
}

define void @_Z3fn2v() local_unnamed_addr {
entry:
  %agg.tmp1 = alloca %class.F, align 8
  %XLValue = alloca %class.F, align 8
  %0 = bitcast %class.F* %XLValue to i8*
  call void @llvm.lifetime.start.p0i8(i64 96, i8* nonnull %0)
  call void @_Z10EmitLValuev(%class.F* nonnull sret(%class.F) %XLValue)
  %1 = bitcast %class.F* %agg.tmp1 to i8*
  call void @llvm.lifetime.start.p0i8(i64 96, i8* nonnull %1)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 nonnull %1, i8* align 8 nonnull %0, i64 96, i1 false)
  call void @_Z3fn11F(%class.F* byval(%class.F) nonnull align 8 %XLValue)
  %call.i = call signext i32 @_ZN1F11isGlobalRegEv(%class.F* nonnull %agg.tmp1)
  call void @llvm.lifetime.end.p0i8(i64 96, i8* nonnull %1)
  call void @llvm.lifetime.end.p0i8(i64 96, i8* nonnull %0)
  ret void
}
