;
; Dump MIR after the IPRA register-usage collector (RegUsageInfoCollector), which
; runs late in codegen (after regalloc, prolog/epilog, pre-emit hooks, etc.).
; Requires -enable-ipra so that pass is scheduled.
; MIR must still show target-flags(x86-plt) on PLT calls.
;
; Note: -print-regusage writes to stderr in PhysicalRegisterUsageInfo::doFinalization
; and is unrelated to this stdout MIR dump; use a separate llc run if you need it.
;
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -enable-ipra -mattr=-avx,-avx2,-avx512f \
; RUN:     -stop-after=RegUsageInfoCollector %s -o - | FileCheck %s


$_ZN7DerivedC2Eiii = comdat any
%struct.VAR = type { i32, i32, i8 }

@goal = local_unnamed_addr global i32 0, align 4
@gstruct_i = global %struct.VAR { i32 10, i32 20, i8 97 }, align 4
@gstruct = local_unnamed_addr global ptr @gstruct_i, align 8

; dso_local so the tail call from @_Z6createiii lowers without PLT; contrasts with
; the comdat constructor's call to @_Z6createiii (not dso_local).
define dso_local void @_Z5dummyiii(i32 noundef %x, i32 noundef %y, i32 noundef %z) local_unnamed_addr #3 {
entry:
  %add = add nsw i32 %z, %x
  store i32 %add, ptr @gstruct_i, align 4  
  %add1 = add nsw i32 %y, 10
  store i32 %add1, ptr getelementptr inbounds (%struct.VAR, ptr @gstruct_i, i64 0, i32 1), align 4 
  store ptr @gstruct_i, ptr @gstruct, align 8 
  ret void
}

; Uses XMM and R8 in assembly (clobbers listed for IPRA / regmask tests).
define dso_local void @dummy_with_xmm_r8() local_unnamed_addr #8 {
entry:
  call void asm sideeffect "movaps %xmm1, %xmm0\0A\09addq $$0, %r8", "~{xmm0},~{xmm1},~{r8},~{dirflag},~{fpsr},~{flags}"()
  ret void
}

; Ensure no PLT flag is call instruction but CustomRegMask from IPRA pass
; CHECK-NOT:   target-flags(x86-plt){{.*}}@_Z5dummyiii
; CHECK:       CALL64pcrel32 @_Z5dummyiii{{.*}}CustomRegMask

; Ensure that Regmask doesnt contain XMM0, XMM1 and R8 used by dummy_with_xmm_r8
; CHECK-NOT:   target-flags(x86-plt){{.*}}@dummy_with_xmm_r8
; CHECK-NOT:   {{,(\\$xmm0|\\$xmm1)[,)]}}
; CHECK-NOT:   {{,\\$r8[,)]}}

define void @_Z6createiii(i32 noundef %x, i32 noundef %y, i32 noundef %z) local_unnamed_addr #4 {
entry:
  call void @_Z5dummyiii(i32 noundef %x, i32 noundef %y, i32 noundef %z)
  call void @dummy_with_xmm_r8()
  %add = add nsw i32 %y, %x
  %add1 = add nsw i32 %add, %z
  store i32 %add1, ptr @goal, align 4
  ret void
}

; Ensure PLT flag is present on call instruction and no CustomRegMask from IPRA pass.
; CHECK:       target-flags(x86-plt) @_Z6createiii
; CHECK-NOT:   CustomRegMask

define linkonce_odr void @_ZN7DerivedC2Eiii(ptr noundef nonnull align 1 dereferenceable(1) %this,
                                            i32 noundef %x, i32 noundef %y, i32 noundef %z) unnamed_addr #7 comdat {
entry:
  tail call void @_Z6createiii(i32 noundef %x, i32 noundef %y, i32 noundef %z)
  ret void
}


