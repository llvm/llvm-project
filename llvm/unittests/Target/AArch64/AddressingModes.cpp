#include "AArch64Subtarget.h"
#include "AArch64TargetMachine.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"

#include "gtest/gtest.h"
#include <initializer_list>
#include <memory>

using namespace llvm;

namespace {

struct AddrMode : public TargetLowering::AddrMode {
  constexpr AddrMode(GlobalValue *GV, int64_t Offs, bool HasBase, int64_t S) {
    BaseGV = GV;
    BaseOffs = Offs;
    HasBaseReg = HasBase;
    Scale = S;
  }
};
struct TestCase {
  AddrMode AM;
  unsigned TypeBits;
  bool Result;
};

const std::initializer_list<TestCase> Tests = {
    // {BaseGV, BaseOffs, HasBaseReg, Scale}, Bits, Result
    {{reinterpret_cast<GlobalValue *>(-1), 0, false, 0}, 64, false},
    {{nullptr, 8, true, 1}, 64, false},
    {{nullptr, 0, false, 2}, 64, true},
    {{nullptr, 0, false, 1}, 64, true},
    {{nullptr, 4, false, 0}, 64, false},

    {{nullptr, 0, true, 1}, 64, true},
    {{nullptr, 0, true, 1}, 32, true},
    {{nullptr, 0, true, 1}, 16, true},
    {{nullptr, 0, true, 1}, 8, true},

    {{nullptr, 0, true, 2}, 64, false},
    {{nullptr, 0, true, 2}, 32, false},
    {{nullptr, 0, true, 2}, 16, true},
    {{nullptr, 0, true, 2}, 8, false},
    {{nullptr, 0, true, 4}, 64, false},
    {{nullptr, 0, true, 4}, 32, true},
    {{nullptr, 0, true, 4}, 16, false},
    {{nullptr, 0, true, 4}, 8, false},

    {{nullptr, 0, true, 8}, 64, true},
    {{nullptr, 0, true, 8}, 32, false},
    {{nullptr, 0, true, 8}, 16, false},
    {{nullptr, 0, true, 8}, 8, false},

    {{nullptr, 0, true, 16}, 64, false},
    {{nullptr, 0, true, 16}, 32, false},
    {{nullptr, 0, true, 16}, 16, false},
    {{nullptr, 0, true, 16}, 8, false},

    {{nullptr, -257, true, 0}, 64, false},
    {{nullptr, -256, true, 0}, 64, true},
    {{nullptr, -255, true, 0}, 64, true},
    {{nullptr, -1, true, 0}, 64, true},
    {{nullptr, 0, true, 0}, 64, true},
    {{nullptr, 1, true, 0}, 64, true},
    {{nullptr, 254, true, 0}, 64, true},
    {{nullptr, 255, true, 0}, 64, true},
    {{nullptr, 256, true, 0}, 64, true},
    {{nullptr, 257, true, 0}, 64, false},
    {{nullptr, 258, true, 0}, 64, false},
    {{nullptr, 259, true, 0}, 64, false},
    {{nullptr, 260, true, 0}, 64, false},
    {{nullptr, 261, true, 0}, 64, false},
    {{nullptr, 262, true, 0}, 64, false},
    {{nullptr, 263, true, 0}, 64, false},
    {{nullptr, 264, true, 0}, 64, true},

    {{nullptr, 4096 * 8 - 8, true, 0}, 64, true},
    {{nullptr, 4096 * 8 - 7, true, 0}, 64, false},
    {{nullptr, 4096 * 8 - 6, true, 0}, 64, false},
    {{nullptr, 4096 * 8 - 5, true, 0}, 64, false},
    {{nullptr, 4096 * 8 - 4, true, 0}, 64, false},
    {{nullptr, 4096 * 8 - 3, true, 0}, 64, false},
    {{nullptr, 4096 * 8 - 2, true, 0}, 64, false},
    {{nullptr, 4096 * 8 - 1, true, 0}, 64, false},
    {{nullptr, 4096 * 8, true, 0}, 64, false},
    {{nullptr, 4096 * 8 + 1, true, 0}, 64, false},
    {{nullptr, 4096 * 8 + 2, true, 0}, 64, false},
    {{nullptr, 4096 * 8 + 3, true, 0}, 64, false},
    {{nullptr, 4096 * 8 + 4, true, 0}, 64, false},
    {{nullptr, 4096 * 8 + 5, true, 0}, 64, false},
    {{nullptr, 4096 * 8 + 6, true, 0}, 64, false},
    {{nullptr, 4096 * 8 + 7, true, 0}, 64, false},
    {{nullptr, 4096 * 8 + 8, true, 0}, 64, false},

    {{nullptr, -257, true, 0}, 32, false},
    {{nullptr, -256, true, 0}, 32, true},
    {{nullptr, -255, true, 0}, 32, true},
    {{nullptr, -1, true, 0}, 32, true},
    {{nullptr, 0, true, 0}, 32, true},
    {{nullptr, 1, true, 0}, 32, true},
    {{nullptr, 254, true, 0}, 32, true},
    {{nullptr, 255, true, 0}, 32, true},
    {{nullptr, 256, true, 0}, 32, true},
    {{nullptr, 257, true, 0}, 32, false},
    {{nullptr, 258, true, 0}, 32, false},
    {{nullptr, 259, true, 0}, 32, false},
    {{nullptr, 260, true, 0}, 32, true},

    {{nullptr, 4096 * 4 - 4, true, 0}, 32, true},
    {{nullptr, 4096 * 4 - 3, true, 0}, 32, false},
    {{nullptr, 4096 * 4 - 2, true, 0}, 32, false},
    {{nullptr, 4096 * 4 - 1, true, 0}, 32, false},
    {{nullptr, 4096 * 4, true, 0}, 32, false},
    {{nullptr, 4096 * 4 + 1, true, 0}, 32, false},
    {{nullptr, 4096 * 4 + 2, true, 0}, 32, false},
    {{nullptr, 4096 * 4 + 3, true, 0}, 32, false},
    {{nullptr, 4096 * 4 + 4, true, 0}, 32, false},

    {{nullptr, -257, true, 0}, 16, false},
    {{nullptr, -256, true, 0}, 16, true},
    {{nullptr, -255, true, 0}, 16, true},
    {{nullptr, -1, true, 0}, 16, true},
    {{nullptr, 0, true, 0}, 16, true},
    {{nullptr, 1, true, 0}, 16, true},
    {{nullptr, 254, true, 0}, 16, true},
    {{nullptr, 255, true, 0}, 16, true},
    {{nullptr, 256, true, 0}, 16, true},
    {{nullptr, 257, true, 0}, 16, false},
    {{nullptr, 258, true, 0}, 16, true},

    {{nullptr, 4096 * 2 - 2, true, 0}, 16, true},
    {{nullptr, 4096 * 2 - 1, true, 0}, 16, false},
    {{nullptr, 4096 * 2, true, 0}, 16, false},
    {{nullptr, 4096 * 2 + 1, true, 0}, 16, false},
    {{nullptr, 4096 * 2 + 2, true, 0}, 16, false},

    {{nullptr, -257, true, 0}, 8, false},
    {{nullptr, -256, true, 0}, 8, true},
    {{nullptr, -255, true, 0}, 8, true},
    {{nullptr, -1, true, 0}, 8, true},
    {{nullptr, 0, true, 0}, 8, true},
    {{nullptr, 1, true, 0}, 8, true},
    {{nullptr, 254, true, 0}, 8, true},
    {{nullptr, 255, true, 0}, 8, true},
    {{nullptr, 256, true, 0}, 8, true},
    {{nullptr, 257, true, 0}, 8, true},

    {{nullptr, 4096 - 2, true, 0}, 8, true},
    {{nullptr, 4096 - 1, true, 0}, 8, true},
    {{nullptr, 4096, true, 0}, 8, false},
    {{nullptr, 4096 + 1, true, 0}, 8, false},

};
} // namespace

TEST(AddressingModes, AddressingModes) {
  LLVMInitializeAArch64TargetInfo();
  LLVMInitializeAArch64Target();
  LLVMInitializeAArch64TargetMC();

  std::string Error;
  auto TT = Triple::normalize("aarch64");
  const Target *T = TargetRegistry::lookupTarget(TT, Error);

  std::unique_ptr<TargetMachine> TM(
      T->createTargetMachine(TT, "generic", "", TargetOptions(), std::nullopt,
                             std::nullopt, CodeGenOptLevel::Default));
  AArch64Subtarget ST(TM->getTargetTriple(), TM->getTargetCPU(),
                      TM->getTargetCPU(), TM->getTargetFeatureString(), *TM,
                      true);

  auto *TLI = ST.getTargetLowering();
  DataLayout DL("e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128");
  LLVMContext Ctx;

  for (const auto &Test : Tests) {
    Type *Typ = Type::getIntNTy(Ctx, Test.TypeBits);
    ASSERT_EQ(TLI->isLegalAddressingMode(DL, Test.AM, Typ, 0), Test.Result);
  }
}
