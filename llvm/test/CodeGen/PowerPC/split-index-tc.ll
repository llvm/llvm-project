; RUN: llc -verify-machineinstrs -mcpu=pwr7 < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

%"class.llvm::MachineOperand" = type { i8, [3 x i8], i64, i64*, i64 }

; Function Attrs: nounwind
define void @_ZN4llvm17ScheduleDAGInstrs14addPhysRegDepsEPNS_5SUnitEj() #0 align 2 {

; If we were able to split out the indexing, the load with update should be
; removed (resulting in a nearly-empty output).
; CHECK-LABEL: @_ZN4llvm17ScheduleDAGInstrs14addPhysRegDepsEPNS_5SUnitEj
; CHECK-NOT: lhzu

entry:
  %0 = load %"class.llvm::MachineOperand"*, %"class.llvm::MachineOperand"** undef, align 8
  br i1 undef, label %_ZNK4llvm14MachineOperand6getRegEv.exit, label %cond.false.i123

cond.false.i123:                                  ; preds = %_ZN4llvm12MachineInstr10getOperandEj.exit
  unreachable

_ZNK4llvm14MachineOperand6getRegEv.exit:          ; preds = %_ZN4llvm12MachineInstr10getOperandEj.exit
  %IsDef.i = getelementptr inbounds %"class.llvm::MachineOperand", %"class.llvm::MachineOperand"* %0, i64 undef, i32 1
  %1 = bitcast [3 x i8]* %IsDef.i to i24*
  %bf.load.i = load i24, i24* %1, align 1
  %2 = and i24 %bf.load.i, 128
  br i1 undef, label %for.cond.cleanup, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %_ZNK4llvm14MachineOperand6getRegEv.exit
  %3 = zext i24 %2 to i32
  br i1 undef, label %cond.false.i134, label %_ZNK4llvm18MCRegAliasIteratordeEv.exit

for.cond.cleanup:                                 ; preds = %_ZNK4llvm14MachineOperand6getRegEv.exit
  br i1 undef, label %_ZNK4llvm14MachineOperand5isDefEv.exit, label %cond.false.i129

cond.false.i129:                                  ; preds = %for.cond.cleanup
  unreachable

_ZNK4llvm14MachineOperand5isDefEv.exit:           ; preds = %for.cond.cleanup
  br i1 undef, label %_ZNK4llvm14MachineOperand6getRegEv.exit247, label %cond.false.i244

cond.false.i134:                                  ; preds = %for.body.lr.ph
  unreachable

_ZNK4llvm18MCRegAliasIteratordeEv.exit:           ; preds = %for.body.lr.ph
  unreachable

cond.false.i244:                                  ; preds = %_ZNK4llvm14MachineOperand5isDefEv.exit
  unreachable

_ZNK4llvm14MachineOperand6getRegEv.exit247:       ; preds = %_ZNK4llvm14MachineOperand5isDefEv.exit
  br i1 undef, label %if.then53, label %if.end55

if.then53:                                        ; preds = %_ZNK4llvm14MachineOperand6getRegEv.exit247
  unreachable

if.end55:                                         ; preds = %_ZNK4llvm14MachineOperand6getRegEv.exit247
  br i1 undef, label %_ZNK4llvm14MachineOperand6isDeadEv.exit262, label %cond.false.i257

cond.false.i257:                                  ; preds = %if.end55
  unreachable

_ZNK4llvm14MachineOperand6isDeadEv.exit262:       ; preds = %if.end55
  %bf.load.i259 = load i24, i24* %1, align 1
  br i1 undef, label %if.then57, label %if.else59

if.then57:                                        ; preds = %_ZNK4llvm14MachineOperand6isDeadEv.exit262
  unreachable

if.else59:                                        ; preds = %_ZNK4llvm14MachineOperand6isDeadEv.exit262
  br i1 undef, label %if.end89, label %if.then62

if.then62:                                        ; preds = %if.else59
  unreachable

if.end89:                                         ; preds = %if.else59
  unreachable
}

attributes #0 = { nounwind }


