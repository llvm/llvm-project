; RUN: llc < %s -mtriple aarch64--

; This tests that the MFI assert in unreachableblockelim pass
; does not trigger

%struct.ngtcp2_crypto_aead = type { ptr, i64 }
%struct.ngtcp2_crypto_aead_ctx = type { ptr }

; Function Attrs: noinline optnone
define internal fastcc void @decrypt_pkt() unnamed_addr #0 !type !0 !type !1 {
entry:
  br i1 false, label %cont, label %trap, !nosanitize !2

trap:                                             ; preds = %entry
  unreachable, !nosanitize !2

cont:                                             ; preds = %entry
  %call = call i32 undef(ptr undef, ptr undef, ptr undef, ptr undef, i64 undef, ptr undef, i64 undef, ptr undef, i64 undef)
  ret void
}

attributes #0 = { noinline optnone }

!0 = !{i64 0, !"_ZTSFlPhPK18ngtcp2_crypto_aeadPKhmS4_mlP16ngtcp2_crypto_kmPFiS_S2_PK22ngtcp2_crypto_aead_ctxS4_mS4_mS4_mEE"}
!1 = !{i64 0, !"_ZTSFlPvPKvS1_mS1_mlS_S_E.generalized"}
!2 = !{}
