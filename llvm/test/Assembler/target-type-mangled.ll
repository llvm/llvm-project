; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; Check support for mangling of target extension types in intrinsics

declare target("a", target("b")) @llvm.ssa.copy.ta_tbtt(target("a", target("b")) returned)
declare target("a", void, i8, 5, 3) @llvm.ssa.copy.ta_isVoid_i8_5_3t(target("a", void, i8, 5, 3) returned)
declare target("b") @llvm.ssa.copy.tbt(target("b") returned)

; CHECK: declare target("a", target("b")) @llvm.ssa.copy.ta_tbtt(target("a", target("b")) returned)
; CHECK: declare target("a", void, i8, 5, 3) @llvm.ssa.copy.ta_isVoid_i8_5_3t(target("a", void, i8, 5, 3) returned)
; CHECK: declare target("b") @llvm.ssa.copy.tbt(target("b") returned)

