; RUN: llc < %s -relocation-model=pic -disable-fp-elim -verify-machineinstrs
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

define void @_ZN12_GLOBAL__N_113SPUAsmPrinter15EmitInstructionEPKN4llvm12MachineInstrE(i8* %this, i8* %MI) nounwind inlinehint align 2 {
entry:
  br i1 undef, label %"3.i", label %"4.i"

"3.i":                                            ; preds = %entry
  unreachable

"4.i":                                            ; preds = %entry
  switch i32 undef, label %_ZN12_GLOBAL__N_113SPUAsmPrinter16printInstructionEPKN4llvm12MachineInstrERNS1_11raw_ostreamE.exit [
    i32 1, label %"5.i"
    i32 2, label %"6.i"
    i32 3, label %"7.i"
    i32 4, label %"8.i"
    i32 5, label %"9.i"
  ]

"5.i":                                            ; preds = %"4.i"
  unreachable

"6.i":                                            ; preds = %"4.i"
  switch i32 undef, label %"11.i" [
    i32 1, label %"12.i"
    i32 2, label %"13.i"
    i32 3, label %_ZN12_GLOBAL__N_113SPUAsmPrinter16printInstructionEPKN4llvm12MachineInstrERNS1_11raw_ostreamE.exit
    i32 4, label %"14.i"
  ]

"7.i":                                            ; preds = %"4.i"
  unreachable

"8.i":                                            ; preds = %"4.i"
  unreachable

"9.i":                                            ; preds = %"4.i"
  unreachable

"11.i":                                           ; preds = %"6.i"
  switch i32 undef, label %"15.i" [
    i32 1, label %"16.i"
    i32 2, label %"17.i"
    i32 3, label %"18.i"
    i32 4, label %"19.i"
    i32 5, label %"20.i"
    i32 6, label %"21.i"
    i32 7, label %"24.i"
    i32 8, label %"27.i"
    i32 9, label %"28.i"
    i32 10, label %"29.i"
    i32 11, label %"30.i"
    i32 12, label %"31.i"
    i32 13, label %"32.i"
    i32 14, label %"39.i"
  ]

"12.i":                                           ; preds = %"6.i"
  unreachable

"13.i":                                           ; preds = %"6.i"
  unreachable

"14.i":                                           ; preds = %"6.i"
  unreachable

"15.i":                                           ; preds = %"11.i"
  unreachable

"16.i":                                           ; preds = %"11.i"
  unreachable

"17.i":                                           ; preds = %"11.i"
  unreachable

"18.i":                                           ; preds = %"11.i"
  unreachable

"19.i":                                           ; preds = %"11.i"
  unreachable

"20.i":                                           ; preds = %"11.i"
  unreachable

"21.i":                                           ; preds = %"11.i"
  br i1 undef, label %"22.i", label %"23.i"

"22.i":                                           ; preds = %"21.i"
  unreachable

"23.i":                                           ; preds = %"21.i"
  unreachable

"24.i":                                           ; preds = %"11.i"
  unreachable

"27.i":                                           ; preds = %"11.i"
  unreachable

"28.i":                                           ; preds = %"11.i"
  unreachable

"29.i":                                           ; preds = %"11.i"
  unreachable

"30.i":                                           ; preds = %"11.i"
  unreachable

"31.i":                                           ; preds = %"11.i"
  unreachable

"32.i":                                           ; preds = %"11.i"
  unreachable

"39.i":                                           ; preds = %"11.i"
  br i1 undef, label %"41.i", label %"40.i"

"40.i":                                           ; preds = %"39.i"
  unreachable

"41.i":                                           ; preds = %"39.i"
  %0 = call i64 @_ZNK4llvm14MachineOperand6getImmEv(i8 undef) nounwind inlinehint ; <i64> [#uses=1]
  %1 = trunc i64 %0 to i16                        ; <i16> [#uses=1]
  br i1 undef, label %"42.i", label %"43.i"

"42.i":                                           ; preds = %"41.i"
  unreachable

"43.i":                                           ; preds = %"41.i"
  %2 = and i16 %1, -16                            ; <i16> [#uses=1]
  %3 = sext i16 %2 to i64                         ; <i64> [#uses=1]
  %4 = call i8 @_ZN4llvm11raw_ostreamlsEl(i8 undef, i64 %3) nounwind ; <i8> [#uses=0]
  unreachable

_ZN12_GLOBAL__N_113SPUAsmPrinter16printInstructionEPKN4llvm12MachineInstrERNS1_11raw_ostreamE.exit: ; preds = %"6.i", %"4.i"
  ret void
}

declare i64 @_ZNK4llvm14MachineOperand6getImmEv(i8) nounwind inlinehint align 2

declare i8 @_ZN4llvm11raw_ostreamlsEl(i8, i64)
