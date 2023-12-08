// RUN: llvm-tblgen -gen-register-info -I %p/../../include %s 2>&1 | FileCheck %s
//
// CHECK-NOT: warning: SubRegIndex Test::subreg_h64 and Test::subreg_h32 compose ambiguously as Test::subreg_hh32 or Test::subreg_h32
// CHECK: warning: SubRegIndex Test::subreg_l64 and Test::subreg_l32 compose ambiguously as Test::subreg_ll32 or Test::subreg_l32

include "llvm/Target/Target.td"

def TestInstrInfo : InstrInfo {
}

def Test : Target {
  let InstructionSet = TestInstrInfo;
}

let Namespace = "Test" in {
  def subreg_l32  : SubRegIndex<32, 0>;
  def subreg_h32  : SubRegIndex<32, 32>;
  def subreg_h64  : SubRegIndex<64, 64>;
  def subreg_l64  : SubRegIndex<64, 0>;
  def subreg_hh32 : ComposedSubRegIndex<subreg_h64, subreg_h32>;
  def subreg_ll32 : ComposedSubRegIndex<subreg_l64, subreg_l32>;
}

class TestReg<string n, list<Register> s> : RegisterWithSubRegs<n, s> {
  let Namespace = "Test";
}

// --------------------------------------------------------------------
// A situation that previously caused the warning about ambiguous
// composition.
//
// The actual subregister actions are:
//   subreg_h64:  { F0Q->F0D V0Q->F0D }
//   subreg_h32:  { F0D->F0S F0Q->F2S V0Q->F0S }
//   composition: { F0Q->F0S V0Q->F0S }  (this is the same as subreg_hh32)
// 
// For the register V0Q, subreg_hh32(V0Q) = subreg_h32(V0Q) = F0S, which
// would be enough to trigger the warning about ambiguous composition.
// However, for F0Q, subreg_hh32(F0Q) = F0S, while subreg_h32(F0Q) = F2S,
// which shows that there two subregister indices are different.
// Make sure that the warning is not emitted in this case.

class FPR32<string n> : TestReg<n, []> {
}

class FPR64<string n, FPR32 high> : TestReg<n, [high]> {
  let SubRegIndices = [subreg_h32];
}

class FPR128<string n, FPR64 high, FPR32 low> : TestReg<n, [high, low]> {
  let SubRegIndices = [subreg_h64, subreg_h32];
}

class VPR128<string n, FPR64 high> : TestReg<n, [high]> {
  let SubRegIndices = [subreg_h64];
}

def F0S : FPR32<"f0s">;
def F1S : FPR32<"f1s">;
def F2S : FPR32<"f2s">;

def F0D : FPR64<"f0d",  F0S>;
def F0Q : FPR128<"f0q", F0D, F2S>;
def V0Q : VPR128<"v0q", F0D>;

def FP32  : RegisterClass<"FP32",  [f32],   32,  (add F0S)>;
def FP64  : RegisterClass<"FP64",  [f64],   64,  (add F0D)>;
def FP128 : RegisterClass<"FP128", [v2f64], 128, (add F0Q)>;
def VP128 : RegisterClass<"VP128", [v2f64], 128, (add V0Q)>;

// --------------------------------------------------------------------
// A situation where the warning is legitimate.
// Make sure that the warning is still displayed.

class GPR32<string n> : TestReg<n, []> {
}

class GPR64<string n, GPR32 low> : TestReg<n, [low]> {
  let SubRegIndices = [subreg_l32];
}

class GPR128<string n, GPR64 low> : TestReg<n, [low]> {
  let SubRegIndices = [subreg_l64];
}

def G0S : GPR32<"g0s">;
def G0D : GPR64<"g0d", G0S>;
def G0Q : GPR128<"g0q", G0D>;

def GP32  : RegisterClass<"GP32",  [i32],   32,  (add G0S)>;
def GP64  : RegisterClass<"GP64",  [i64],   64,  (add G0D)>;
def GP128 : RegisterClass<"GP128", [v2i64], 128, (add G0Q)>;
