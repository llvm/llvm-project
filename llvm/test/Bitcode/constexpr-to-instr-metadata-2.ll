; RUN: llvm-dis -expand-constant-exprs < %S/Inputs/constexpr-to-instr-metadata-2.bc | FileCheck %s

; CHECK-LABEL: define void @_ZN4alsa3pcm3PCM17hw_params_current17hf1c237aece2f69c4E() {
; CHECK: call void @llvm.dbg.value(metadata ptr undef, metadata !4, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 64)), !dbg !14

; CHECK-LABEL: define void @_ZN4alsa3pcm8HwParams3any17h02a64cfc85ce8a66E() {
; CHECK: call void @llvm.dbg.value(metadata ptr undef, metadata !23, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 64)), !dbg !28
