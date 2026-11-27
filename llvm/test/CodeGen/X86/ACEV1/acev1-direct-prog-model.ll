; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+acev1,+amx-tile \
; RUN:     -stop-after=finalize-isel -o - | FileCheck %s
;
; The programming model is recorded per function and per extension, and is read
; off the pseudo opcode during ISel. The ACE DirectReg intrinsics used to select
; the AMX pseudos, which recorded them against the AMX model and left a function
; containing nothing but ACE looking like an AMX one.

; Every ACE DirectReg intrinsic that shares a real instruction with AMX, in one
; function: a single one of them selecting an AMX pseudo would show up as an AMX
; model here.
define void @ace_direct(i32 %i, ptr %out) {
; CHECK-LABEL: name:            ace_direct
; CHECK:         amxProgModel:    None
; CHECK:         aceProgModel:    DirectReg
  call void @llvm.x86.acev1.tilezero(i8 1)
  %a = call <16 x float> @llvm.x86.acev1.tcvtrowd2ps(i8 1, i32 %i)
  store volatile <16 x float> %a, ptr %out, align 64
  %b = call <32 x bfloat> @llvm.x86.acev1.tcvtrowps2bf16h(i8 1, i32 %i)
  store volatile <32 x bfloat> %b, ptr %out, align 64
  %c = call <32 x bfloat> @llvm.x86.acev1.tcvtrowps2bf16l(i8 1, i32 %i)
  store volatile <32 x bfloat> %c, ptr %out, align 64
  %d = call <32 x half> @llvm.x86.acev1.tcvtrowps2phh(i8 1, i32 %i)
  store volatile <32 x half> %d, ptr %out, align 64
  %e = call <32 x half> @llvm.x86.acev1.tcvtrowps2phl(i8 1, i32 %i)
  store volatile <32 x half> %e, ptr %out, align 64
  %f = call <16 x i32> @llvm.x86.acev1.tilemovrowextract(i8 1, i32 %i)
  store volatile <16 x i32> %f, ptr %out, align 64
  ret void
}

; The AMX spelling of a shared instruction still belongs to AMX. Same target,
; so only the intrinsic can tell these two functions apart.
define void @amx_direct() {
; CHECK-LABEL: name:            amx_direct
; CHECK:         amxProgModel:    DirectReg
; CHECK:         aceProgModel:    None
  call void @llvm.x86.tilezero(i8 1)
  ret void
}

; Both kinds is fine if the compiler configures neither; the user switches.
define void @both_direct() {
; CHECK-LABEL: name:            both_direct
; CHECK:         amxProgModel:    DirectReg
; CHECK:         aceProgModel:    DirectReg
  call void @llvm.x86.tilezero(i8 1)
  call void @llvm.x86.acev1.tilezero(i8 1)
  ret void
}

declare void @llvm.x86.acev1.tilezero(i8)
declare <16 x float> @llvm.x86.acev1.tcvtrowd2ps(i8, i32)
declare <32 x bfloat> @llvm.x86.acev1.tcvtrowps2bf16h(i8, i32)
declare <32 x bfloat> @llvm.x86.acev1.tcvtrowps2bf16l(i8, i32)
declare <32 x half> @llvm.x86.acev1.tcvtrowps2phh(i8, i32)
declare <32 x half> @llvm.x86.acev1.tcvtrowps2phl(i8, i32)
declare <16 x i32> @llvm.x86.acev1.tilemovrowextract(i8, i32)
declare void @llvm.x86.tilezero(i8)
