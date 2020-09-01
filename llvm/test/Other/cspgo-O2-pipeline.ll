; Test CSGen pass in CSPGO.
; RUN: llvm-profdata merge %S/Inputs/cspgo-noncs.proftext -o %t-noncs.profdata
; RUN: llvm-profdata merge %S/Inputs/cspgo-cs.proftext -o %t-cs.profdata
; RUN: opt -enable-new-pm=0 -O2 -debug-pass=Structure -pgo-kind=pgo-instr-use-pipeline -profile-file='%t-noncs.profdata' -cspgo-kind=cspgo-instr-gen-pipeline -cs-profilegen-file=alloc %s 2>&1 |FileCheck %s --check-prefixes=CSGENDEFAULT
; CSGENDEFAULT: PGOInstrumentationUse
; CSGENDEFAULT: PGOInstrumentationGenCreateVar
; CSGENDEFAULT: PGOInstrumentationGen

; Test CSUse pass in CSPGO.
; RUN: opt -enable-new-pm=0 -O2 -debug-pass=Structure -pgo-kind=pgo-instr-use-pipeline -profile-file='%t-cs.profdata' -cspgo-kind=cspgo-instr-use-pipeline %s 2>&1 |FileCheck %s --check-prefixes=CSUSEDEFAULT
; CSUSEDEFAULT: PGOInstrumentationUse
; CSUSEDEFAULT-NOT: PGOInstrumentationGenCreateVar
; CSUSEDEFAULT: PGOInstrumentationUse
