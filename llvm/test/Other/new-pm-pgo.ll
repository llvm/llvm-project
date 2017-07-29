; RUN: opt -debug-pass-manager -passes='default<O2>' -pgo-kind=new-pm-pgo-instr-gen-pipeline -profile-file='temp' %s 2>&1 |FileCheck %s --check-prefixes=GEN
; RUN: llvm-profdata merge %S/Inputs/new-pm-pgo.proftext -o %t.profdata
; RUN: opt -debug-pass-manager -passes='default<O2>' -pgo-kind=new-pm-pgo-instr-use-pipeline -profile-file='%t.profdata' %s 2>&1 |FileCheck %s --check-prefixes=USE
; RUN: opt -debug-pass-manager -passes='default<O2>' -pgo-kind=new-pm-pgo-sample-use-pipeline -profile-file='%S/Inputs/new-pm-pgo.prof' %s 2>&1 |FileCheck %s --check-prefixes=SAMPLE_USE
; RUN: opt -debug-pass-manager -passes='default<O2>' -new-pm-debug-info-for-profiling %s 2>&1 |FileCheck %s --check-prefixes=SAMPLE_GEN
;
; GEN: Running pass: PGOInstrumentationGen
; USE: Running pass: PGOInstrumentationUse
; SAMPLE_USE: Running pass: ModuleToFunctionPassAdaptor<llvm::AddDiscriminatorsPass>
; SAMPLE_USE: Running pass: SampleProfileLoaderPass
; SAMPLE_GEN: Running pass: ModuleToFunctionPassAdaptor<llvm::AddDiscriminatorsPass>

define void @foo() {
  ret void
}
