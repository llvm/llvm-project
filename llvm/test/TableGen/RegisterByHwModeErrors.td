// RUN: rm -rf %t && split-file %s %t
// RUN: not llvm-tblgen --gen-asm-matcher -I %t -I %p/../../include -I %S \
// RUN:    %t/bad-regclass.td -o /dev/null 2>&1 | FileCheck %t/bad-regclass.td --implicit-check-not="error:"
// RUN: not llvm-tblgen --gen-asm-matcher -I %t -I %p/../../include -I %S \
// RUN:    %t/bad-regclass-by-mode.td -o /dev/null 2>&1 | FileCheck %t/bad-regclass-by-mode.td --implicit-check-not="error:"
// RUN: not llvm-tblgen --gen-asm-matcher -I %t -I %p/../../include -I %S \
// RUN:    %t/bad-not-regclass.td -o /dev/null 2>&1 | FileCheck %t/bad-not-regclass.td --implicit-check-not="error:"
// RUN: not llvm-tblgen --gen-asm-matcher -I %t -I %p/../../include -I %S \
// RUN:    %t/duplicate-entry.td -o /dev/null 2>&1 | FileCheck %t/duplicate-entry.td --implicit-check-not="error:"
// RUN: not llvm-tblgen --gen-dag-isel -I %t -I %p/../../include -I %S \
// RUN:    %t/isel-not-supported-yet.td -o /dev/null 2>&1 | FileCheck %t/isel-not-supported-yet.td --implicit-check-not="error:"

//--- bad-regclass.td
include "Common/RegisterByHwModeCommon.td"
def BadReg : RegisterByHwMode<XRegs, [XPtr32, XPtr64, YPtr32, YPtr64], [X0, X0, X0, Y0]>;
// CHECK: [[#@LINE-1]]:5: error: Register Y0 for HwMode YPtr64 is not a member of register class XRegs

// Need to define an instruction that uses BadReg to get the diagnostic
def : InstAlias<"test $src", (TEST_XREG BadReg, XRegs:$src)>;
def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }


//--- bad-regclass-by-mode.td
include "Common/RegisterByHwModeCommon.td"
def BadReg : RegisterByHwMode<PtrRC, [XPtr32, XPtr64, YPtr32, YPtr64], [X0, X0, X0, X0]>;
// CHECK: [[#@LINE-1]]:5: error: Register X0 for HwMode YPtr32 is not a member of register class YRegs

// Need to define an instruction that uses BadReg to get the diagnostic
def : InstAlias<"test $src", (TEST_PTRREG BadReg, PtrRegOperand:$src)>;
def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }


//--- bad-not-regclass.td
include "Common/RegisterByHwModeCommon.td"
def FakeRegClass : RegisterClassLike;
def BadReg : RegisterByHwMode<FakeRegClass, [XPtr32, XPtr64, YPtr32, YPtr64], [X0, X0, X0, X0]>;
// CHECK: [[#@LINE-1]]:5: error: FakeRegClass is not a known RegisterClass!
// CHECK: [[#@LINE-3]]:5: note: FakeRegClass defined here

// Need to define an instruction that uses BadReg to get the diagnostic
def : InstAlias<"test $src", (TEST_XREG BadReg, XRegs:$src)>;
def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }


//--- duplicate-entry.td
include "Common/RegisterByHwModeCommon.td"
/// We should get an error if we accidentally use the same mode twice:
def BadReg : RegisterByHwMode<XRegs, [XPtr32, XPtr64, XPtr32, YPtr64], [X0, X0, X0, Y0]>;
// CHECK: [[#@LINE-1]]:5: error: duplicate Register for HwMode DefaultMode: X0

// Need to define an instruction that uses BadReg to get the diagnostic
def : InstAlias<"test $src", (TEST_XREG BadReg, XRegs:$src)>;
def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }

//--- isel-not-supported-yet.td
include "Common/RegisterByHwModeCommon.td"
/// Output for -gen-dag-isel is not supported yet, check that we get an
/// error instead of crashing.
def int_with_null_reg : Intrinsic<[llvm_i64_ty], [], [IntrNoMem, IntrWillReturn]>;
def : Pat<(int_with_null_reg), (TEST_PTRREG NullReg)>;
// TODO: We don't track source locations of the DAG nodes, so the error location is bad.
// CHECK: RegisterByHwModeCommon.td:55:5: error: RegisterByHwMode in SelectionDAG patterns not yet supported!

def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }
