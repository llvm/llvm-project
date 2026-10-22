; Bitcode compatibility test for speculative devirt resolutions in
; thin-lto summaries. Checks that older bitcode without the speculative
; resolutions are still properly parsed.

; RUN: llvm-dis < %S/Inputs/thinlto-summary-speculative-devirt-14.bc | FileCheck %s --check-prefix=CHECK-DIS
; RUN: llvm-bcanalyzer -dump %S/Inputs/thinlto-summary-speculative-devirt-14.bc | FileCheck %s --check-prefix=CHECK-BCAN

; CHECK-BCAN: <GLOBALVAL_SUMMARY_BLOCK
; CHECK-BCAN: <VERSION op0=14/>

; CHECK-DIS: ^2 = typeid: (name: "WPDType", summary: (typeTestRes: (kind: allOnes, sizeM1BitWidth: 0), wpdResolutions: ((offset: 0, wpdRes: (kind: singleImpl, singleImplName: "f")), (offset: 8, wpdRes: (kind: indir, resByArg: (args: (1), byArg: (kind: uniformRetVal, info: 7)))), (offset: 16, wpdRes: (kind: branchFunnel)), (offset: 24, wpdRes: (kind: singleImpl, singleImplName: "g")))))
