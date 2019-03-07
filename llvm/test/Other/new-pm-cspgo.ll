; Test CSGen pass in CSPGO.
; RUN: llvm-profdata merge %S/Inputs/cspgo-noncs.proftext -o %t-noncs.profdata
; RUN: llvm-profdata merge %S/Inputs/cspgo-cs.proftext -o %t-cs.profdata
; RUN: opt -debug-pass-manager -passes='default<O2>' -pgo-kind=pgo-instr-use-pipeline -profile-file='%t-noncs.profdata' -cspgo-kind=cspgo-instr-gen-pipeline -cs-profilegen-file=alloc %s 2>&1 |FileCheck %s --check-prefixes=CSGENDEFAULT
; RUN: opt -debug-pass-manager -passes='thinlto-pre-link<O2>' -pgo-kind=pgo-instr-use-pipeline -profile-file='%t-noncs.profdata' -cspgo-kind=cspgo-instr-gen-pipeline -cs-profilegen-file=alloc %s 2>&1 |FileCheck %s --check-prefixes=CSGENPRELINK
; RUN: opt -debug-pass-manager -passes='thinlto<O2>' -pgo-kind=pgo-instr-use-pipeline -profile-file='%t-noncs.profdata' -cspgo-kind=cspgo-instr-gen-pipeline -cs-profilegen-file=alloc %s 2>&1 |FileCheck %s --check-prefixes=CSGENLTO
; RUN: opt -debug-pass-manager -passes='lto-pre-link<O2>' -pgo-kind=pgo-instr-use-pipeline -profile-file='%t-noncs.profdata' -cspgo-kind=cspgo-instr-gen-pipeline -cs-profilegen-file=alloc %s 2>&1 |FileCheck %s --check-prefixes=CSGENPRELINK
; RUN: opt -debug-pass-manager -passes='lto<O2>' -pgo-kind=pgo-instr-use-pipeline -profile-file='%t-noncs.profdata' -cspgo-kind=cspgo-instr-gen-pipeline -cs-profilegen-file=alloc %s 2>&1 |FileCheck %s --check-prefixes=CSGENLTO
; CSGENDEFAULT: Running pass: PGOInstrumentationUse
; CSGENDEFAULT: Running pass: PGOInstrumentationGenCreateVar
; CSGENDEFAULT: Running pass: PGOInstrumentationGen
; CSGENPRELINK: Running pass: PGOInstrumentationUse
; CSGENPRELINK: Running pass: PGOInstrumentationGenCreateVar
; CSGENPRELINK-NOT: Running pass: PGOInstrumentationGen
; CSGENLTO-NOT: Running pass: PGOInstrumentationUse
; CSGENLTO-NOT: Running pass: PGOInstrumentationGenCreateVar
; CSGENLTO: Running pass: PGOInstrumentationGen

; Test CSUse pass in CSPGO.
; RUN: opt -debug-pass-manager -passes='default<O2>' -pgo-kind=pgo-instr-use-pipeline -profile-file='%t-cs.profdata' -cspgo-kind=cspgo-instr-use-pipeline %s 2>&1 |FileCheck %s --check-prefixes=CSUSEDEFAULT
; RUN: opt -debug-pass-manager -passes='thinlto-pre-link<O2>' -pgo-kind=pgo-instr-use-pipeline -profile-file='%t-cs.profdata' -cspgo-kind=cspgo-instr-use-pipeline %s 2>&1 |FileCheck %s --check-prefixes=CSUSEPRELINK
; RUN: opt -debug-pass-manager -passes='thinlto<O2>' -pgo-kind=pgo-instr-use-pipeline -profile-file='%t-cs.profdata' -cspgo-kind=cspgo-instr-use-pipeline %s 2>&1 |FileCheck %s --check-prefixes=CSUSELTO
; RUN: opt -debug-pass-manager -passes='lto-pre-link<O2>' -pgo-kind=pgo-instr-use-pipeline -profile-file='%t-noncs.profdata' -cspgo-kind=cspgo-instr-use-pipeline %s 2>&1 |FileCheck %s --check-prefixes=CSUSEPRELINK
; RUN: opt -debug-pass-manager -passes='lto<O2>' -pgo-kind=pgo-instr-use-pipeline -profile-file='%t-cs.profdata' -cspgo-kind=cspgo-instr-use-pipeline %s 2>&1 |FileCheck %s --check-prefixes=CSUSELTO
; CSUSEDEFAULT: Running pass: PGOInstrumentationUse
; CSUSEDEFAULT-NOT: Running pass: PGOInstrumentationGenCreateVar
; CSUSEDEFAULT: Running pass: PGOInstrumentationUse
; CSUSEPRELINK: Running pass: PGOInstrumentationUse
; CSUSEPRELINK-NOT: Running pass: PGOInstrumentationGenCreateVar
; CSUSEPRELINK-NOT: Running pass: PGOInstrumentationUse
; CSUSELTO: Running pass: PGOInstrumentationUse
; CSUSELTO-NOT: Running pass: PGOInstrumentationUse
