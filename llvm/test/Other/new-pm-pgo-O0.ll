; RUN: opt -debug-pass-manager -passes='default<O0>' -pgo-kind=pgo-instr-gen-pipeline -profile-file='temp' %s 2>&1 |FileCheck %s --check-prefixes=GEN
; RUN: llvm-profdata merge %S/Inputs/new-pm-pgo.proftext -o %t.profdata
; RUN: opt -debug-pass-manager -passes='default<O0>' -pgo-kind=pgo-instr-use-pipeline -profile-file='%t.profdata' %s 2>&1 |FileCheck %s --check-prefixes=USE_DEFAULT,USE
; RUN: opt -debug-pass-manager -passes='thinlto-pre-link<O0>' -pgo-kind=pgo-instr-use-pipeline -profile-file='%t.profdata' %s 2>&1 \
; RUN:     |FileCheck %s --check-prefixes=USE_PRE_LINK,USE
; RUN: opt -debug-pass-manager -passes='lto-pre-link<O0>' -pgo-kind=pgo-instr-use-pipeline -profile-file='%t.profdata' %s 2>&1 \
; RUN:     |FileCheck %s --check-prefixes=USE_PRE_LINK,USE
; RUN: opt -debug-pass-manager -passes='thinlto<O0>' -pgo-kind=pgo-instr-use-pipeline -profile-file='%t.profdata' %s 2>&1 \
; RUN:     |FileCheck %s --check-prefixes=USE_POST_LINK,USE
; RUN: opt -debug-pass-manager -passes='lto<O0>' -pgo-kind=pgo-instr-use-pipeline -profile-file='%t.profdata' %s 2>&1 \
; RUN:     |FileCheck %s --check-prefixes=USE_POST_LINK,USE

;
; GEN: Running pass: PGOInstrumentationGen
; USE_DEFAULT: Running pass: PGOInstrumentationUse
; USE_PRE_LINK: Running pass: PGOInstrumentationUse
; USE_POST_LINK-NOT: Running pass: PGOInstrumentationUse
; USE-NOT: Running pass: PGOIndirectCallPromotion
; USE-NOT: Running pass: PGOMemOPSizeOpt

define void @foo() {
  ret void
}
