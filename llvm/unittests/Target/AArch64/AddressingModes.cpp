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
  constexpr AddrMode(GlobalValue *GV, int64_t Offs, bool HasBase, int64_t S,
                     int64_t SOffs = 0) {
    BaseGV = GV;
    BaseOffs = Offs;
    HasBaseReg = HasBase;
    Scale = S;
    ScalableOffset = SOffs;
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

struct SVETestCase {
  AddrMode AM;
  unsigned TypeBits;
  unsigned NumElts;
  bool Result;
};

const std::initializer_list<SVETestCase> SVETests = {
    // {BaseGV, BaseOffs, HasBaseReg, Scale, SOffs}, EltBits, Count, Result
    // Test immediate range -- [-8,7] vector's worth.
    // <vscale x 16 x i8>, increment by one vector
    {{nullptr, 0, true, 0, 16}, 8, 16, true},
    // <vscale x 4 x i32>, increment by eight vectors
    {{nullptr, 0, true, 0, 128}, 32, 4, false},
    // <vscale x 8 x i16>, increment by seven vectors
    {{nullptr, 0, true, 0, 112}, 16, 8, true},
    // <vscale x 2 x i64>, decrement by eight vectors
    {{nullptr, 0, true, 0, -128}, 64, 2, true},
    // <vscale x 16 x i8>, decrement by nine vectors
    {{nullptr, 0, true, 0, -144}, 8, 16, false},

    // Half the size of a vector register, but allowable with extending
    // loads and truncating stores
    // <vscale x 8 x i8>, increment by three vectors
    {{nullptr, 0, true, 0, 24}, 8, 8, true},

    // Test invalid types or offsets
    // <vscale x 5 x i32>, increment by one vector (base size > 16B)
    {{nullptr, 0, true, 0, 20}, 32, 5, false},
    // <vscale x 8 x i16>, increment by half a vector
    {{nullptr, 0, true, 0, 8}, 16, 8, false},
    // <vscale x 3 x i8>, increment by 3 vectors (non-power-of-two)
    {{nullptr, 0, true, 0, 9}, 8, 3, false},

    // Scalable and fixed offsets
    // <vscale x 16 x i8>, increment by 32 then decrement by vscale x 16
    {{nullptr, 32, true, 0, -16}, 8, 16, false},
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

  for (const auto &SVETest : SVETests) {
    Type *Ty = VectorType::get(Type::getIntNTy(Ctx, SVETest.TypeBits),
                               ElementCount::getScalable(SVETest.NumElts));
    ASSERT_EQ(TLI->isLegalAddressingMode(DL, SVETest.AM, Ty, 0),
              SVETest.Result);
  }
}
