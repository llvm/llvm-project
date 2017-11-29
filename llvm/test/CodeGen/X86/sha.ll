; RUN: llc < %s -mattr=+sha -mtriple=x86_64-unknown-unknown | FileCheck %s
; RUN: not llc < %s -mtriple=x86_64-unknown-unknown

declare <4 x i32> @llvm.x86.sha1rnds4(<4 x i32>, <4 x i32>, i8) nounwind readnone

define <4 x i32> @test_sha1rnds4rr(<4 x i32> %a, <4 x i32> %b) nounwind uwtable {
entry:
  %0 = tail call <4 x i32> @llvm.x86.sha1rnds4(<4 x i32> %a, <4 x i32> %b, i8 3)
  ret <4 x i32> %0
  ; CHECK: test_sha1rnds4rr
  ; CHECK: sha1rnds4 $3, %xmm1, %xmm0
}

define <4 x i32> @test_sha1rnds4rm(<4 x i32> %a, <4 x i32>* %b) nounwind uwtable {
entry:
  %0 = load <4 x i32>, <4 x i32>* %b
  %1 = tail call <4 x i32> @llvm.x86.sha1rnds4(<4 x i32> %a, <4 x i32> %0, i8 3)
  ret <4 x i32> %1
  ; CHECK: test_sha1rnds4rm
  ; CHECK: sha1rnds4 $3, (%rdi), %xmm0
}

declare <4 x i32> @llvm.x86.sha1nexte(<4 x i32>, <4 x i32>) nounwind readnone

define <4 x i32> @test_sha1nexterr(<4 x i32> %a, <4 x i32> %b) nounwind uwtable {
entry:
  %0 = tail call <4 x i32> @llvm.x86.sha1nexte(<4 x i32> %a, <4 x i32> %b)
  ret <4 x i32> %0
  ; CHECK: test_sha1nexterr
  ; CHECK: sha1nexte %xmm1, %xmm0
}

define <4 x i32> @test_sha1nexterm(<4 x i32> %a, <4 x i32>* %b) nounwind uwtable {
entry:
  %0 = load <4 x i32>, <4 x i32>* %b
  %1 = tail call <4 x i32> @llvm.x86.sha1nexte(<4 x i32> %a, <4 x i32> %0)
  ret <4 x i32> %1
  ; CHECK: test_sha1nexterm
  ; CHECK: sha1nexte (%rdi), %xmm0
}

declare <4 x i32> @llvm.x86.sha1msg1(<4 x i32>, <4 x i32>) nounwind readnone

define <4 x i32> @test_sha1msg1rr(<4 x i32> %a, <4 x i32> %b) nounwind uwtable {
entry:
  %0 = tail call <4 x i32> @llvm.x86.sha1msg1(<4 x i32> %a, <4 x i32> %b)
  ret <4 x i32> %0
  ; CHECK: test_sha1msg1rr
  ; CHECK: sha1msg1 %xmm1, %xmm0
}

define <4 x i32> @test_sha1msg1rm(<4 x i32> %a, <4 x i32>* %b) nounwind uwtable {
entry:
  %0 = load <4 x i32>, <4 x i32>* %b
  %1 = tail call <4 x i32> @llvm.x86.sha1msg1(<4 x i32> %a, <4 x i32> %0)
  ret <4 x i32> %1
  ; CHECK: test_sha1msg1rm
  ; CHECK: sha1msg1 (%rdi), %xmm0
}

declare <4 x i32> @llvm.x86.sha1msg2(<4 x i32>, <4 x i32>) nounwind readnone

define <4 x i32> @test_sha1msg2rr(<4 x i32> %a, <4 x i32> %b) nounwind uwtable {
entry:
  %0 = tail call <4 x i32> @llvm.x86.sha1msg2(<4 x i32> %a, <4 x i32> %b)
  ret <4 x i32> %0
  ; CHECK: test_sha1msg2rr
  ; CHECK: sha1msg2 %xmm1, %xmm0
}

define <4 x i32> @test_sha1msg2rm(<4 x i32> %a, <4 x i32>* %b) nounwind uwtable {
entry:
  %0 = load <4 x i32>, <4 x i32>* %b
  %1 = tail call <4 x i32> @llvm.x86.sha1msg2(<4 x i32> %a, <4 x i32> %0)
  ret <4 x i32> %1
  ; CHECK: test_sha1msg2rm
  ; CHECK: sha1msg2 (%rdi), %xmm0
}

declare <4 x i32> @llvm.x86.sha256rnds2(<4 x i32>, <4 x i32>, <4 x i32>) nounwind readnone

define <4 x i32> @test_sha256rnds2rr(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c) nounwind uwtable {
entry:
  %0 = tail call <4 x i32> @llvm.x86.sha256rnds2(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c)
  ret <4 x i32> %0
  ; CHECK: test_sha256rnds2rr
  ; CHECK: movaps %xmm0, [[xmm_TMP1:%xmm[1-9][0-9]?]]
  ; CHECK: movaps %xmm2, %xmm0
  ; CHECK: sha256rnds2 %xmm0, %xmm1, [[xmm_TMP1]]
}

define <4 x i32> @test_sha256rnds2rm(<4 x i32> %a, <4 x i32>* %b, <4 x i32> %c) nounwind uwtable {
entry:
  %0 = load <4 x i32>, <4 x i32>* %b
  %1 = tail call <4 x i32> @llvm.x86.sha256rnds2(<4 x i32> %a, <4 x i32> %0, <4 x i32> %c)
  ret <4 x i32> %1
  ; CHECK: test_sha256rnds2rm
  ; CHECK: movaps %xmm0, [[xmm_TMP2:%xmm[1-9][0-9]?]]
  ; CHECK: movaps %xmm1, %xmm0
  ; CHECK: sha256rnds2 %xmm0, (%rdi), [[xmm_TMP2]]
}

declare <4 x i32> @llvm.x86.sha256msg1(<4 x i32>, <4 x i32>) nounwind readnone

define <4 x i32> @test_sha256msg1rr(<4 x i32> %a, <4 x i32> %b) nounwind uwtable {
entry:
  %0 = tail call <4 x i32> @llvm.x86.sha256msg1(<4 x i32> %a, <4 x i32> %b)
  ret <4 x i32> %0
  ; CHECK: test_sha256msg1rr
  ; CHECK: sha256msg1 %xmm1, %xmm0
}

define <4 x i32> @test_sha256msg1rm(<4 x i32> %a, <4 x i32>* %b) nounwind uwtable {
entry:
  %0 = load <4 x i32>, <4 x i32>* %b
  %1 = tail call <4 x i32> @llvm.x86.sha256msg1(<4 x i32> %a, <4 x i32> %0)
  ret <4 x i32> %1
  ; CHECK: test_sha256msg1rm
  ; CHECK: sha256msg1 (%rdi), %xmm0
}

declare <4 x i32> @llvm.x86.sha256msg2(<4 x i32>, <4 x i32>) nounwind readnone

define <4 x i32> @test_sha256msg2rr(<4 x i32> %a, <4 x i32> %b) nounwind uwtable {
entry:
  %0 = tail call <4 x i32> @llvm.x86.sha256msg2(<4 x i32> %a, <4 x i32> %b)
  ret <4 x i32> %0
  ; CHECK: test_sha256msg2rr
  ; CHECK: sha256msg2 %xmm1, %xmm0
}

define <4 x i32> @test_sha256msg2rm(<4 x i32> %a, <4 x i32>* %b) nounwind uwtable {
entry:
  %0 = load <4 x i32>, <4 x i32>* %b
  %1 = tail call <4 x i32> @llvm.x86.sha256msg2(<4 x i32> %a, <4 x i32> %0)
  ret <4 x i32> %1
  ; CHECK: test_sha256msg2rm
  ; CHECK: sha256msg2 (%rdi), %xmm0
}
