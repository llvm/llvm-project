#include "llvm/Support/Yk.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"

using namespace llvm;

bool YkExtendedLLVMBBAddrMapSection;
namespace {
struct CreateYkExtendedLLVMBBAddrMapSectionParser {
  static void *call() {
    return new cl::opt<bool, true>(
        "yk-extended-llvmbbaddrmap-section",
        cl::desc("Use the extended Yk `.llvmbbaddrmap` section format"),
        cl::NotHidden, cl::location(YkExtendedLLVMBBAddrMapSection));
  }
};
} // namespace
static ManagedStatic<cl::opt<bool, true>,
                     CreateYkExtendedLLVMBBAddrMapSectionParser>
    YkExtendedLLVMBBAddrMapSectionParser;

bool YkStackMapOffsetFix;
namespace {
struct CreateYkStackMapOffsetFixParser {
  static void *call() {
    return new cl::opt<bool, true>(
        "yk-stackmap-offset-fix",
        cl::desc(
            "Apply a fix to stackmaps that corrects the reported instruction "
            "offset in the presence of calls. (deprecated by "
            "yk-stackmap-spillreloads-fix)"),
        cl::NotHidden, cl::location(YkStackMapOffsetFix));
  }
};
} // namespace
static ManagedStatic<cl::opt<bool, true>, CreateYkStackMapOffsetFixParser>
    YkStackMapOffsetFixParser;

bool YkStackMapAdditionalLocs;
namespace {
struct CreateYkStackMapAdditionalLocsParser {
  static void *call() {
    return new cl::opt<bool, true>(
        "yk-stackmap-add-locs",
        cl::desc("Encode additional locations for registers into stackmaps."),
        cl::NotHidden, cl::location(YkStackMapAdditionalLocs));
  }
};
} // namespace
static ManagedStatic<cl::opt<bool, true>, CreateYkStackMapAdditionalLocsParser>
    YkStackMapAdditionalLocsParser;

bool YkStackmapsSpillReloadsFix;
namespace {
struct CreateYkStackmapsSpillReloadsFixParser {
  static void *call() {
    return new cl::opt<bool, true>(
        "yk-stackmap-spillreloads-fix",
        cl::desc(
            "Revert stackmaps and its operands after the register allocator "
            "has emitted spill reloads."),
        cl::NotHidden, cl::location(YkStackmapsSpillReloadsFix));
  }
};
} // namespace
static ManagedStatic<cl::opt<bool, true>,
                     CreateYkStackmapsSpillReloadsFixParser>
    YkStackmapsSpillFixParser;

bool YkOptNoneAfterIRPasses;
namespace {
struct CreateYkOptNoneAfterIRPassesParser {
  static void *call() {
    return new cl::opt<bool, true>(
        "yk-optnone-after-ir-passes",
        cl::desc(
            "Apply `optnone` to all functions prior to instruction selection."),
        cl::NotHidden, cl::location(YkOptNoneAfterIRPasses));
  }
};
} // namespace
static ManagedStatic<cl::opt<bool, true>,
  CreateYkOptNoneAfterIRPassesParser> YkOptNoneAfterIRPassesParser;

bool YkEmbedIR;
namespace {
struct CreateYkEmbedIRParser {
  static void *call() {
    return new cl::opt<bool, true>(
        "yk-embed-ir",
        cl::desc(
            "Embed Yk IR into the binary."),
        cl::NotHidden, cl::location(YkEmbedIR));
  }
};
} // namespace
static ManagedStatic<cl::opt<bool, true>, CreateYkEmbedIRParser> YkEmbedIRParser;

bool YkDontOptFuncABI;
namespace {
struct CreateYkDontOptFuncABIParser {
  static void *call() {
    return new cl::opt<bool, true>(
        "yk-dont-opt-func-abi",
        cl::desc(
            "Don't change the ABIs of functions during optimisation"),
        cl::NotHidden, cl::location(YkDontOptFuncABI));
  }
};
} // namespace
static ManagedStatic<cl::opt<bool, true>, CreateYkDontOptFuncABIParser> YkDontOptFuncABIParser;

void llvm::initYkOptions() {
  *YkExtendedLLVMBBAddrMapSectionParser;
  *YkStackMapOffsetFixParser;
  *YkStackMapAdditionalLocsParser;
  *YkStackmapsSpillFixParser;
  *YkOptNoneAfterIRPassesParser;
  *YkEmbedIRParser;
  *YkDontOptFuncABIParser;
}
