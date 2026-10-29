// Test that TableGen emits an error when RegClassByHwMode cannot be resolved
// due to missing or contradictory predicates, leading to a type mismatch.
// RUN: not llvm-tblgen --gen-compress-inst-emitter -I %p/../../include -I %S %s -DERROR_UNRESOLVED 2>&1 | FileCheck %s --check-prefix=UNRESOLVED
// RUN: not llvm-tblgen --gen-compress-inst-emitter -I %p/../../include -I %S %s -DERROR_CONTRADICTORY_PAT 2>&1 | FileCheck %s --check-prefix=CONTRADICTORY_PAT
// RUN: not llvm-tblgen --gen-compress-inst-emitter -I %p/../../include -I %S %s -DERROR_WRONG_PRED 2>&1 | FileCheck %s --check-prefix=WRONG_PRED
// RUN: not llvm-tblgen --gen-compress-inst-emitter -I %p/../../include -I %S %s -DERROR_CONTRADICTORY_MODE 2>&1 | FileCheck %s --check-prefix=CONTRADICTORY_MODE
// RUN: not llvm-tblgen --gen-compress-inst-emitter -I %p/../../include -I %S %s -DERROR_INCOMPATIBLE_HWMODE 2>&1 | FileCheck %s --check-prefix=INCOMPATIBLE_HWMODE
// RUN: not llvm-tblgen --gen-compress-inst-emitter -I %p/../../include -I %S %s -DERROR_NON_ASM_PRED_MODE 2>&1 | FileCheck %s --check-prefix=NON_ASM_PRED_MODE

include "Common/RegClassByHwModeCommon.td"

def FeaturePtr64 : SubtargetFeature<"ptr64", "IsPtr64", "true", "Ptr64 mode">;

def IsPtr64 : Predicate<"Subtarget->isPtr64()">,
              AssemblerPredicate<(all_of FeaturePtr64)>;
def NotPtr64 : Predicate<"!Subtarget->isPtr64()">,
               AssemblerPredicate<(all_of (not FeaturePtr64))>;

defvar Ptr32 = DefaultMode;
#ifdef ERROR_NON_ASM_PRED_MODE
/// NonAsmPred has no AssemblerPredicate, so NonAsmMode is only active when both
/// IsPtr64 and NonAsmPred hold. DefaultMode can still be active when IsPtr64
/// is true and NonAsmPred is false, so a pattern predicated only on IsPtr64
/// cannot uniquely resolve PtrRC to YRegs.
def NonAsmPred : Predicate<"Subtarget->someOtherCheck()">;
def NonAsmMode : HwMode<[IsPtr64, NonAsmPred]>;
def PtrRC : RegClassByHwMode<[Ptr32, NonAsmMode], [XRegs, YRegs]>;
#else
def Ptr64 : HwMode<[IsPtr64]>;
#ifdef ERROR_CONTRADICTORY_MODE
// BadMode has contradictory predicates, so it can never be active.
def BadMode : HwMode<[IsPtr64, NotPtr64]>;
def PtrRC : RegClassByHwMode<[Ptr32, BadMode], [XRegs, YRegs]>;
#else
def PtrRC : RegClassByHwMode<[Ptr32, Ptr64], [XRegs, YRegs]>;
#endif
def OtherRC : RegClassByHwMode<[Ptr32, Ptr64], [YRegs, XRegs]>;
#endif

def X_MOV : TestInstruction {
  let OutOperandList = (outs XRegs:$dst);
  let InOperandList = (ins XRegs:$src);
  let AsmString = "x_mov $dst, $src";
  let opcode = 0;
}

def Y_MOV : TestInstruction {
  let OutOperandList = (outs YRegs:$dst);
  let InOperandList = (ins YRegs:$src);
  let AsmString = "y_mov $dst, $src";
  let opcode = 5;
}

def PTR_MOV : TestInstruction {
  let OutOperandList = (outs PtrRC:$dst);
  let InOperandList = (ins PtrRC:$src);
  let AsmString = "ptr_mov $dst, $src";
  let opcode = 3;
}

def PTR_MOV_SMALL : TestInstruction {
  let OutOperandList = (outs PtrRC:$dst);
  let InOperandList = (ins PtrRC:$src);
  let AsmString = "ptr_mov.small $dst, $src";
  let opcode = 4;
  let Size = 1;
}

#ifdef ERROR_UNRESOLVED
// X_MOV uses XRegs, but PTR_MOV_SMALL uses PtrRC.
// Since there are no predicates, PtrRC cannot be resolved.
def : CompressPat<(X_MOV XRegs:$dst, XRegs:$src),
                  (PTR_MOV_SMALL XRegs:$dst, XRegs:$src)>;
// UNRESOLVED: {{.*}}RegClassByHwModeCompressPatError.td:[[@LINE-2]]:{{[0-9]+}}: error: Error in Dag '{{.*}}'. Operand 'dst' has type 'XRegs' which does not match the type 'PtrRC' in the corresponding instruction operand!
#endif

#ifdef ERROR_CONTRADICTORY_PAT
// Pattern has contradictory predicates, so no modes can be active.
let Predicates = [IsPtr64, NotPtr64] in
def : CompressPat<(X_MOV XRegs:$dst, XRegs:$src),
                  (PTR_MOV_SMALL XRegs:$dst, XRegs:$src)>;
// CONTRADICTORY_PAT: {{.*}}RegClassByHwModeCompressPatError.td:[[@LINE-2]]:{{[0-9]+}}: error: Error in Dag '{{.*}}'. Operand 'dst' has type 'XRegs' which does not match the type 'PtrRC' in the corresponding instruction operand!
#endif

#ifdef ERROR_WRONG_PRED
// Pattern has NotPtr64 predicate, so PtrRC resolves to XRegs.
// But Y_MOV uses YRegs, which is incompatible with XRegs.
let Predicates = [NotPtr64] in
def : CompressPat<(Y_MOV YRegs:$dst, YRegs:$src),
                  (PTR_MOV_SMALL YRegs:$dst, YRegs:$src)>;
// WRONG_PRED: {{.*}}RegClassByHwModeCompressPatError.td:[[@LINE-2]]:{{[0-9]+}}: error: Error in Dag '{{.*}}'. Operand 'dst' has type 'YRegs' which does not match the type 'PtrRC' in the corresponding instruction operand!
#endif

#ifdef ERROR_CONTRADICTORY_MODE
// Pattern has IsPtr64 predicate, which should resolve PtrRC to YRegs (from BadMode).
// But BadMode is incompatible because it has contradictory predicates.
// So PtrRC cannot be resolved.
let Predicates = [IsPtr64] in
def : CompressPat<(Y_MOV YRegs:$dst, YRegs:$src),
                  (PTR_MOV_SMALL YRegs:$dst, YRegs:$src)>;
// CONTRADICTORY_MODE: {{.*}}RegClassByHwModeCompressPatError.td:[[@LINE-2]]:{{[0-9]+}}: error: Error in Dag '{{.*}}'. Operand 'dst' has type 'YRegs' which does not match the type 'PtrRC' in the corresponding instruction operand!
#endif

#ifdef ERROR_INCOMPATIBLE_HWMODE
// OtherRC has YRegs for Ptr32 and XRegs for Ptr64.
// PtrRC has XRegs for Ptr32 and YRegs for Ptr64.
// They are incompatible because they don't have subclass relationship in all modes.
def : CompressPat<(PTR_MOV OtherRC:$dst, OtherRC:$src),
                  (PTR_MOV_SMALL PtrRC:$dst, PtrRC:$src)>;
// INCOMPATIBLE_HWMODE: {{.*}}RegClassByHwModeCompressPatError.td:[[@LINE-2]]:{{[0-9]+}}: error: Error in Dag '{{.*}}'. Operand 'dst' has type 'OtherRC' which does not match the type 'PtrRC' in the corresponding instruction operand!
#endif

#ifdef ERROR_NON_ASM_PRED_MODE
/// Pattern has IsPtr64 predicate, but NonAsmMode also requires NonAsmPred, so
/// DefaultMode (XRegs) is still possible when IsPtr64 holds.
let Predicates = [IsPtr64] in
def : CompressPat<(Y_MOV YRegs:$dst, YRegs:$src),
                  (PTR_MOV_SMALL YRegs:$dst, YRegs:$src)>;
// NON_ASM_PRED_MODE: {{.*}}RegClassByHwModeCompressPatError.td:[[@LINE-2]]:{{[0-9]+}}: error: Error in Dag '{{.*}}'. Operand 'dst' has type 'YRegs' which does not match the type 'PtrRC' in the corresponding instruction operand!
#endif

def MyTargetISA : InstrInfo;
def MyTargetAsmWriter : AsmWriter {
  let PassSubtarget = 1;
}
def MyTarget : Target {
  let InstructionSet = MyTargetISA;
  let AssemblyWriters = [MyTargetAsmWriter];
}
