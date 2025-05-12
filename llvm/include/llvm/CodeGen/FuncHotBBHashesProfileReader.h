#ifndef LLVM_CODEGEN_FuncHotBBHashesProfileReader_H
#define LLVM_CODEGEN_FuncHotBBHashesProfileReader_H

#include "llvm/ADT/SmallString.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/MemoryBuffer.h"

namespace llvm {

struct HotBBInfo {
    uint64_t BBHash;
    uint64_t Freq;
};

class FuncHotBBHashesProfileReader : public ImmutablePass {
public:
    static char ID;

    FuncHotBBHashesProfileReader(const std::string PropellerProfile);

    FuncHotBBHashesProfileReader();

    StringRef getPassName() const override {
        return "Basic Block Frequency with Hash Profile Reader";
    }

    // return a vector of hit BB hashes for a function.
    std::pair<bool, SmallVector<HotBBInfo, 4>>
    getHotBBInfosForFunction(StringRef FuncName) const;

    // Reads the profile for matching functions.
    bool doInitialization(Module &M) override;

private:
    std::unique_ptr<MemoryBuffer> MBuf;

    StringRef getAliasName(StringRef FuncName) const {
        auto R = FuncAliasMap.find(FuncName);
        return R == FuncAliasMap.end() ? FuncName : R->second;
    }

    // Reads the basic block frequency with hash profile for functions in this module.
    Error ReadProfile();
    
    // Profile file path.
    std::string PropellerFilePath;

    // Some functions have alias names. We use this map to find the main alias
    // name for which we have mapping in ProgramBBClusterInfo.
    StringMap<StringRef> FuncAliasMap;

    // record the frequency of basic block, 
    // the basic block is represented by its hash.
    StringMap<SmallVector<HotBBInfo, 4>> FuncToHotBBHashes;
};

ImmutablePass *
createFuncHotBBHashesProfileReaderPass(const std::string PropellerProfile);

} // namespace llvm

#endif // LLVM_CODEGEN_FuncHotBBHashesProfileReader_H