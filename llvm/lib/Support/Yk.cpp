#include "llvm/Support/CommandLine.h"

using namespace llvm;

bool YkAllocLLVMBCSection;
static cl::opt<bool, true> YkAllocLLVMBCSectionParser(
    "yk-alloc-llvmbc-section", cl::desc("Make the `.llvmbc` section loadable"),
    cl::NotHidden, cl::location(YkAllocLLVMBCSection));

bool YkAllocLLVMBBAddrMapSection;
static cl::opt<bool, true> YkAllocLLVMBBAddrMapSectionParser(
    "yk-alloc-llvmbbaddrmap-section",
    cl::desc("Make the `.llvmbbaddrmap` section loadable"), cl::NotHidden,
    cl::location(YkAllocLLVMBBAddrMapSection));

bool YkExtendedLLVMBBAddrMapSection;
static cl::opt<bool, true> YkExtendedLLVMBBAddrMapSectionParser(
    "yk-extended-llvmbbaddrmap-section",
    cl::desc("Use the extended Yk `.llvmbbaddrmap` section format"),
    cl::NotHidden, cl::location(YkExtendedLLVMBBAddrMapSection));
