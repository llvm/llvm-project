; ACE v1 Greedy Register Allocation Test
; Tests that tile registers are allocated correctly in a separate pass
; Note: ACE v1 uses TILEMOVROW to extract tile data (not TILESTORED)
;
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+acev1,+avx512f,+avx512bf16 -verify-machineinstrs -stop-after x86-tile-config | FileCheck %s

; Test basic tile register allocation for ACE
define void @test_acev1_ra(ptr %out, <32 x bfloat> %zmm_a, <32 x bfloat> %zmm_b) nounwind {
  ; CHECK-LABEL: name: test_acev1_ra
  ; CHECK: PLDTILECFGV
  ; CHECK: PTILEZEROV
  ; CHECK: PTOP2BF16PSV
  ; CHECK: PTILEMOVROWrtiV
entry:
  %c = tail call x86_amx @llvm.x86.tilezero.internal(i16 16, i16 64)

  %d = tail call x86_amx @llvm.x86.top2bf16ps.internal(i16 16, i16 64, i16 64, x86_amx %c, <32 x bfloat> %zmm_a, <32 x bfloat> %zmm_b)

  ; Extract using TILEMOVROW (ACE v1 pattern)
  %row0 = call <16 x i32> @llvm.x86.tilemovrow.internal(i16 16, i16 64, x86_amx %d, i32 0)
  store volatile <16 x i32> %row0, ptr %out, align 64

  ret void
}

; Test multiple tile operations requiring allocation
define void @test_acev1_ra_multi(ptr %out, <32 x bfloat> %zmm_a, <32 x bfloat> %zmm_b) nounwind {
  ; CHECK-LABEL: name: test_acev1_ra_multi
  ; CHECK: PLDTILECFGV
  ; CHECK: PTILEZEROV
  ; CHECK: PTILEZEROV
  ; CHECK: PTOP2BF16PSV
entry:
  %c1 = tail call x86_amx @llvm.x86.tilezero.internal(i16 16, i16 64)
  %c2 = tail call x86_amx @llvm.x86.tilezero.internal(i16 16, i16 64)

  %d1 = tail call x86_amx @llvm.x86.top2bf16ps.internal(i16 16, i16 64, i16 64, x86_amx %c1, <32 x bfloat> %zmm_a, <32 x bfloat> %zmm_b)
  %d2 = tail call x86_amx @llvm.x86.top2bf16ps.internal(i16 16, i16 64, i16 64, x86_amx %c2, <32 x bfloat> %zmm_a, <32 x bfloat> %zmm_b)

  ; Extract using TILEMOVROW
  %row0 = call <16 x i32> @llvm.x86.tilemovrow.internal(i16 16, i16 64, x86_amx %d2, i32 0)
  store volatile <16 x i32> %row0, ptr %out, align 64

  ret void
}

; Test integer outer products register allocation
define void @test_acev1_ra_int(ptr %out, <64 x i8> %a_i8) nounwind {
  ; CHECK-LABEL: name: test_acev1_ra_int
  ; CHECK: PLDTILECFGV
  ; CHECK: PTILEZEROV
  ; CHECK: PTOP4BSSDV
  ; CHECK: PTOP4BUUDV
entry:
  %c = tail call x86_amx @llvm.x86.tilezero.internal(i16 16, i16 64)

  %d1 = tail call x86_amx @llvm.x86.top4bssd.internal(i16 16, i16 64, i16 64, x86_amx %c, <64 x i8> %a_i8, <64 x i8> %a_i8)
  %d2 = tail call x86_amx @llvm.x86.top4buud.internal(i16 16, i16 64, i16 64, x86_amx %d1, <64 x i8> %a_i8, <64 x i8> %a_i8)

  ; Extract using TILEMOVROW
  %row0 = call <16 x i32> @llvm.x86.tilemovrow.internal(i16 16, i16 64, x86_amx %d2, i32 0)
  store volatile <16 x i32> %row0, ptr %out, align 64

  ret void
}

; Test register allocation with control flow
define void @test_acev1_ra_branch(ptr %out, i32 %cond, <32 x bfloat> %zmm_a, <32 x bfloat> %zmm_b) nounwind {
  ; CHECK-LABEL: name: test_acev1_ra_branch
  ; CHECK: PLDTILECFGV
entry:
  %c = tail call x86_amx @llvm.x86.tilezero.internal(i16 16, i16 64)

  %icmp = icmp eq i32 %cond, 0
  br i1 %icmp, label %then, label %else

then:
  %d1 = tail call x86_amx @llvm.x86.top2bf16ps.internal(i16 16, i16 64, i16 64, x86_amx %c, <32 x bfloat> %zmm_a, <32 x bfloat> %zmm_b)
  br label %merge

else:
  %d2 = tail call x86_amx @llvm.x86.top2bf16ps.internal(i16 16, i16 64, i16 64, x86_amx %c, <32 x bfloat> %zmm_b, <32 x bfloat> %zmm_a)
  br label %merge

merge:
  %result = phi x86_amx [ %d1, %then ], [ %d2, %else ]

  ; Extract using TILEMOVROW
  %row0 = call <16 x i32> @llvm.x86.tilemovrow.internal(i16 16, i16 64, x86_amx %result, i32 0)
  store volatile <16 x i32> %row0, ptr %out, align 64

  ret void
}

declare x86_amx @llvm.x86.tilezero.internal(i16, i16)
declare x86_amx @llvm.x86.top2bf16ps.internal(i16, i16, i16, x86_amx, <32 x bfloat>, <32 x bfloat>)
declare x86_amx @llvm.x86.top4bssd.internal(i16, i16, i16, x86_amx, <64 x i8>, <64 x i8>)
declare x86_amx @llvm.x86.top4buud.internal(i16, i16, i16, x86_amx, <64 x i8>, <64 x i8>)
declare <16 x i32> @llvm.x86.tilemovrow.internal(i16, i16, x86_amx, i32)
