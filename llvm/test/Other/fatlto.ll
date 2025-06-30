; RUN: opt -debug-pass-manager -passes='fatlto-pre-link<O2>' -disable-output %s 2>&1 | FileCheck %s
; RUN: opt -debug-pass-manager -passes='fatlto-pre-link<O2;thinlto>' -disable-output %s 2>&1 | FileCheck %s --check-prefixes=CHECK,THINLTO

; CHECK: Running pass: EmbedBitcodePass on [module]
; THINLTO: Running analysis: ModuleSummaryIndexAnalysis on [module]
; CHECK: Running pass: FatLtoCleanup on [module]
; CHECK-NEXT: Running pass: LowerTypeTestsPass on [module]
