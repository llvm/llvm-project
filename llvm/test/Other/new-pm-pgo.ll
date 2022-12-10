; RUN: opt -debug-pass-manager -passes='default<O2>' -pgo-kind=pgo-instr-gen-pipeline -profile-file='temp' %s 2>&1 |FileCheck %s --check-prefixes=GEN
; RUN: llvm-profdata merge %S/Inputs/new-pm-pgo.proftext -o %t.profdata
; RUN: opt -debug-pass-manager -passes='default<O2>' -pgo-kind=pgo-instr-use-pipeline -profile-file='%t.profdata' %s 2>&1 |FileCheck %s --check-prefixes=USE
; RUN: opt -debug-pass-manager -passes='thinlto-pre-link<O2>' -pgo-kind=pgo-instr-use-pipeline -profile-file='%t.profdata' %s 2>&1 |FileCheck %s --check-prefixes=USE
; RUN: opt -debug-pass-manager -passes='thinlto<O2>' -pgo-kind=pgo-instr-use-pipeline -profile-file='%t.profdata' %s 2>&1 |FileCheck %s --check-prefixes=USE_POST_LINK
; RUN: opt -debug-pass-manager -passes='default<O2>' -hot-cold-split -pgo-kind=pgo-instr-use-pipeline -profile-file='%t.profdata' %s 2>&1 |FileCheck %s --check-prefixes=USE --check-prefixes=SPLIT
; RUN: opt -debug-pass-manager -passes='default<O2>' -pgo-kind=pgo-sample-use-pipeline -profile-file='%S/Inputs/new-pm-pgo.prof' %s 2>&1 \
; RUN:     |FileCheck %s --check-prefixes=SAMPLE_USE,SAMPLE_USE_O
; RUN: opt -debug-pass-manager -passes='thinlto-pre-link<O2>' -pgo-kind=pgo-sample-use-pipeline -profile-file='%S/Inputs/new-pm-pgo.prof' %s 2>&1 \
; RUN:     |FileCheck %s --check-prefixes=SAMPLE_USE,SAMPLE_USE_PRE_LINK
; RUN: opt -debug-pass-manager -passes='thinlto<O2>' -pgo-kind=pgo-sample-use-pipeline -profile-file='%S/Inputs/new-pm-pgo.prof' %s 2>&1 \
; RUN:     |FileCheck %s --check-prefixes=SAMPLE_USE,SAMPLE_USE_POST_LINK
; RUN: opt -debug-pass-manager -passes='default<O2>' -debug-info-for-profiling %s 2>&1 |FileCheck %s --check-prefixes=SAMPLE_GEN
;
; GEN: Running pass: PGOInstrumentationGen
; USE: Running pass: PGOInstrumentationUse
; USE: Running pass: PGOIndirectCallPromotion
; USE: Running pass: PGOMemOPSizeOpt
; USE_POST_LINK: Running pass: PGOIndirectCallPromotion
; USE_POST_LINK: Running pass: PGOMemOPSizeOpt
; SAMPLE_USE_O: Running pass: AddDiscriminatorsPass
; SAMPLE_USE_PRE_LINK: Running pass: AddDiscriminatorsPass
; SAMPLE_USE: Running pass: LowerExpectIntrinsicPass
; SAMPLE_USE: Running pass: SimplifyCFGPass
; SAMPLE_USE: Running pass: SROAPass
; SAMPLE_USE: Running pass: EarlyCSEPass
; SAMPLE_USE_POST_LINK: Running pass: InstCombinePass
; SAMPLE_USE: Running pass: SampleProfileLoaderPass
; SAMPLE_USE_O: Running pass: PGOIndirectCallPromotion
; SAMPLE_USE_POST_LINK-NOT: Running pass: GlobalOptPass
; SAMPLE_USE_POST_LINK: Running pass: PGOIndirectCallPromotion
; SAMPLE_USE_PRE_LINK-NOT: Running pass: PGOIndirectCallPromotion
; SAMPLE_GEN: Running pass: AddDiscriminatorsPass
; SPLIT: Running pass: HotColdSplittingPass

define void @foo() {
  ret void
}
