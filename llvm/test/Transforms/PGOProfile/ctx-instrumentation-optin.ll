; REQUIRES: linux
;
; RUN: opt -O2 -debug-pass-manager -S -pgo-kind=pgo-instr-gen-pipeline -profile-file='temp' \
; RUN:  < %s  2>&1 | FileCheck %s --check-prefixes=COMMON,PGO
; RUN: opt -O2 -debug-pass-manager -S -pgo-kind=pgo-instr-gen-pipeline -profile-file='temp' \
; RUN:  -profile-context-root=something < %s 2>&1 | FileCheck %s --check-prefixes=COMMON,CTXPROF

; COMMON:   Running pass: PGOInstrumentationGen
; COMMON:   Invalidating analysis: InnerAnalysisManagerProxy<FunctionAnalysisManager, Module>
; COMMON:   Invalidating analysis: LazyCallGraphAnalysis
; COMMON:   Invalidating analysis: InnerAnalysisManagerProxy<CGSCCAnalysisManager, Module>
; CTXPROF:  Running pass: AssignGUIDPass
; CTXPROF:  Running pass: NoinlineNonPrevailing
; COMMON:   Running analysis: InnerAnalysisManagerProxy<FunctionAnalysisManager, Module>
; PGO:      Running pass: InstrProfilingLoweringPass
; CTXPROF:  Running pass: PGOCtxProfLoweringPass
