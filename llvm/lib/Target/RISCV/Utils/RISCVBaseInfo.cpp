#include "RISCVBaseInfo.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
namespace RISCVSysReg {
#define GET_SysRegsList_IMPL
#include "RISCVGenSystemOperands.inc"
} // namespace RISCVSysReg

namespace RISCVABI {
ABI computeTargetABI(const Triple &TT, FeatureBitset FeatureBits,
                     StringRef ABIName) {
  auto TargetABI = StringSwitch<ABI>(ABIName)
                       .Case("ilp32", ABI_ILP32)
                       .Case("ilp32f", ABI_ILP32F)
                       .Case("ilp32d", ABI_ILP32D)
                       .Case("ilp32e", ABI_ILP32E)
                       .Case("lp64", ABI_LP64)
                       .Case("lp64f", ABI_LP64F)
                       .Case("lp64d", ABI_LP64D)
                       .Default(ABI_Unknown);

  if (!ABIName.empty() && TargetABI == ABI_Unknown) {
    errs()
        << "'" << ABIName
        << "' is not a recognized ABI for this target (ignoring target-abi)\n";
  } else if (ABIName.startswith("ilp32") && TT.isArch64Bit()) {
    errs() << "32-bit ABIs are not supported for 64-bit targets (ignoring "
              "target-abi)\n";
    TargetABI = ABI_Unknown;
  } else if (ABIName.startswith("lp64") && !TT.isArch64Bit()) {
    errs() << "64-bit ABIs are not supported for 32-bit targets (ignoring "
              "target-abi)\n";
    TargetABI = ABI_Unknown;
  } else if (ABIName.endswith("f") && !FeatureBits[RISCV::FeatureStdExtF]) {
    errs() << "Hard-float 'f' ABI can't be used for a target that "
              "doesn't support the F instruction set extension (ignoring "
              "target-abi)\n";
    TargetABI = ABI_Unknown;
  } else if (ABIName.endswith("d") && !FeatureBits[RISCV::FeatureStdExtD]) {
    errs() << "Hard-float 'd' ABI can't be used for a target that "
              "doesn't support the D instruction set extension (ignoring "
              "target-abi)\n";
    TargetABI = ABI_Unknown;
  }

  // For now, default to the ilp32/lp64 if no explicit ABI is given or an
  // invalid/unrecognised string is given. In the future, it might be worth
  // changing this to default to ilp32f/lp64f and ilp32d/lp64d when hardware
  // support for floating point is present.
  if (TargetABI == ABI_Unknown) {
    TargetABI = TT.isArch64Bit() ? ABI_LP64 : ABI_ILP32;
  }

  return TargetABI;
}
} // namespace RISCVABI
} // namespace llvm
