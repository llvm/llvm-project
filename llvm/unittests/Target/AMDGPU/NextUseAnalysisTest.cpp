//===- llvm/unittests/Target/AMDGPU/NextUseAnalysisTest.cpp ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// ============================================================================
// NEXTUSE ANALYSIS UNIT TESTING STRATEGY
// ============================================================================
//
// TEST FRAMEWORK WORKFLOW:
// This test suite parses existing LIT tests from
// llvm/test/CodeGen/AMDGPU/NextUseAnalysis/ and runs NextUseAnalysis once per
// MachineFunction, then queries next-use distances for each instruction in each
// block, using the parsed CHECK patterns as expected results.
//
// COVERAGE EVICTION CONCEPT:
// Coverage eviction occurs when a full register use eliminates sub-register
// uses of the same virtual register because the full register use covers all
// sub-register lanes. This happens during the insert() operation when uses are
// stored.
//
// REGISTER USE ORDERING AND CONTROL FLOW SEMANTICS:
// Different LaneMask (full register/sub-register) uses of the same register are
// stored in distance-increasing order. This ordering reveals the control flow
// structure:
//
// 1. FullRegUse -> SubRegUse: Uses converged at predecessor from independent
// paths.
//    Both are reachable through different branches. Without coverage eviction,
//    this indicates paths diverged and merged at this block.
//
// 2. SubRegUse -> FullRegUse: Sub-register use precedes full register use in
//    linear or fully post-dominated control flow. The full register use would
//    have evicted sub-register uses if they were on the same path.
//
// SPILLING DECISION OPTIMIZATION:
// For optimal spilling decisions, we need the CLOSEST distance among all valid
// paths. Without branch prediction or PGO data, we treat all execution paths as
// equally likely, so the optimal choice is always the nearest next-use.
//
// FULL vs SUB-REGISTER PRECEDENCE LOGIC:
// - If full register use has closer distance than sub-register use:
//   Use full register distance for ALL sub-register queries of that VReg
// - If sub-register use is closer: Use sub-register distance
// - Rationale: Full register use implies ALL sub-register components are used,
//   so it provides the true "next use" for any sub-register query
//
// This strategy ensures unit tests reflect real compiler optimization decisions
// while validating the correctness of the NextUseAnalysis algorithm.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUTargetMachine.h"
#include "AMDGPUUnitTests.h"
#include "GCNSubtarget.h"
#include "llvm/CodeGen/MIRParser/MIRParser.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionAnalysisManager.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/Timer.h"
#include "gtest/gtest.h"
#include <filesystem>
#include <fstream>
#include <regex>

#include "AMDGPUNextUseAnalysis.h"

using namespace llvm;

namespace {

// Helper wrapper to store analysis results for unit testing
class NextUseAnalysisTestWrapper : public AMDGPUNextUseAnalysisWrapper {
public:
  static std::unique_ptr<NextUseResult> Captured;

  bool runOnMachineFunction(MachineFunction &MF) override {
    bool Changed = AMDGPUNextUseAnalysisWrapper::runOnMachineFunction(MF);
    // Store the result for unit test access
    Captured = std::make_unique<NextUseResult>(std::move(getNU()));
    return Changed;
  }
};

std::unique_ptr<NextUseResult> NextUseAnalysisTestWrapper::Captured;


class NextUseAnalysisTestBase : public testing::Test {
protected:
  std::unique_ptr<LLVMContext> Ctx;
  std::unique_ptr<Module> M;
  std::unique_ptr<const GCNTargetMachine> TM;
  MachineModuleInfo *MMI = nullptr;

  // Add TRI and MRI as class members - initialized from first MachineFunction
  const SIRegisterInfo *TRI = nullptr;
  const MachineRegisterInfo *MRI = nullptr;

  // Helper function to convert MachineInstr to string representation
  std::string machineInstrToString(const MachineInstr &MI) {
    std::string Str;
    raw_string_ostream OS(Str);
    MI.print(OS, /*IsStandalone=*/false, /*SkipOpers=*/false,
             /*SkipDebugLoc=*/true, /*AddNewLine=*/false);
    return OS.str();
  }

  inline unsigned getSubRegIndexForLaneMask(LaneBitmask Mask,
                                            const SIRegisterInfo *TRI) {
    for (unsigned Idx = 1; Idx < TRI->getNumSubRegIndices(); ++Idx) {
      if (TRI->getSubRegIndexLaneMask(Idx) == Mask)
        return Idx;
    }
    return AMDGPU::NoRegister;
  }

  // Helper function to map sub-register names to LaneBitmask values using LLVM
  // API
  LaneBitmask getSubRegLaneMask(const std::string &subRegName, Register VReg,
                                const SIRegisterInfo *TRI,
                                const MachineRegisterInfo *MRI) {
    if (subRegName.empty()) {
      return MRI->getMaxLaneMaskForVReg(VReg); // Proper full register mask
    }

    for (unsigned i = 1, e = TRI->getNumSubRegIndices(); i < e; ++i) {
      const char *Name = TRI->getSubRegIndexName(i);
      if (Name && subRegName == Name) {
        return TRI->getSubRegIndexLaneMask(i);
      }
    }

    // If unknown sub-register, default to full register
    return MRI->getMaxLaneMaskForVReg(VReg);
  }

  // Helper function to parse expected distances from CHECK patterns
  llvm::DenseMap<VRegMaskPair, unsigned>
  parseExpectedDistances(const std::vector<std::string> &checkPatterns,
                         const std::string &instrString,
                         const SIRegisterInfo *TRI,
                         const MachineRegisterInfo *MRI) {
    llvm::DenseMap<VRegMaskPair, unsigned> expectedDistances;

    // Clean up the instruction string - remove newlines and extra spaces
    std::string cleanInstrString = instrString;
    cleanInstrString.erase(
        std::remove(cleanInstrString.begin(), cleanInstrString.end(), '\n'),
        cleanInstrString.end());

    // Find the CHECK pattern that matches this instruction
    for (size_t i = 0; i < checkPatterns.size(); ++i) {
      const std::string &pattern = checkPatterns[i];

      // Look for instruction patterns like "CHECK: Instr: %9:vgpr_32 = COPY
      // killed $vgpr1"
      if (pattern.find("CHECK: Instr:") != std::string::npos) {
        // Extract the instruction part after "Instr: "
        size_t instrPos = pattern.find("Instr: ") + 7;
        std::string checkInstr = pattern.substr(instrPos);

        // Matching: compare the actual instruction content
        // Remove leading/trailing whitespace from both
        checkInstr =
            std::regex_replace(checkInstr, std::regex("^\\s+|\\s+$"), "");
        std::string trimmedInstrString =
            std::regex_replace(cleanInstrString, std::regex("^\\s+|\\s+$"), "");

        // Extract the core instruction (everything after the first space)
        size_t firstSpace = checkInstr.find(' ');
        size_t instrFirstSpace = trimmedInstrString.find(' ');

        if (firstSpace != std::string::npos &&
            instrFirstSpace != std::string::npos) {
          std::string checkCore = checkInstr.substr(firstSpace + 1);
          std::string instrCore =
              trimmedInstrString.substr(instrFirstSpace + 1);

          // Match if the core instruction parts are identical
          if (checkCore == instrCore) {
            // Also verify the destination register matches
            std::regex destRegex(R"(^(%\d+:[a-zA-Z_0-9]+)\s*=)");
            std::smatch checkMatch, instrMatch;

            if (std::regex_search(checkInstr, checkMatch, destRegex) &&
                std::regex_search(trimmedInstrString, instrMatch, destRegex) &&
                checkMatch[1].str() == instrMatch[1].str()) {

              // Found exact match, look for subsequent Vreg distance patterns
              // Track full register uses by virtual register to apply precedence
              llvm::DenseMap<Register, unsigned> fullRegDistances;
              
              for (size_t j = i + 1; j < checkPatterns.size(); ++j) {
                const std::string &distPattern = checkPatterns[j];

                // Stop if we hit another instruction or non-Vreg pattern
                if (distPattern.find("CHECK: Instr:") != std::string::npos ||
                    distPattern.find("CHECK: ---") != std::string::npos ||
                    distPattern.find("CHECK-LABEL:") != std::string::npos ||
                    distPattern.find("CHECK: Block End Distances:") !=
                        std::string::npos) {
                  break;
                }

                // Enhanced regex to capture sub-register patterns like "CHECK:
                // Vreg: %15:sub0[ 22 ]" Group 1: register number, Group 2: full
                // sub-register part (optional), Group 3: sub-register name,
                // Group 4: distance
                std::regex vregRegex(
                    R"(CHECK:\s*Vreg:\s*%(\d+)(:([a-zA-Z_0-9]+))?\[\s*(\d+)\s*\])");
                std::smatch vregMatch;

                if (std::regex_search(distPattern, vregMatch, vregRegex) &&
                    vregMatch.size() >= 5) {
                  unsigned regNum = std::stoul(vregMatch[1].str());
                  std::string subRegName =
                      vregMatch[3].str(); // May be empty for full register
                  unsigned distance = std::stoul(vregMatch[4].str());                  Register VReg = Register::index2VirtReg(regNum);
                  LaneBitmask mask = getSubRegLaneMask(subRegName, VReg, TRI, MRI);
                  
                  // Check if this is a full register use
                  if (subRegName.empty() || mask == MRI->getMaxLaneMaskForVReg(VReg)) {
                    // This is a full register use - record it for precedence
                    fullRegDistances[VReg] = distance;
                    VRegMaskPair VMP(VReg, mask);
                    expectedDistances[VMP] = distance;
                  } else {
                    // This is a sub-register use  
                    // Check if we already have a full register use for this VReg with closer distance
                    auto FullIt = fullRegDistances.find(VReg);
                    if (FullIt != fullRegDistances.end() && FullIt->second < distance) {
                      // Full register use exists and is closer - use its distance
                      VRegMaskPair VMP(VReg, mask);
                      expectedDistances[VMP] = FullIt->second;
                    } else {
                      // No full register use or sub-register is closer - use sub-register distance
                      VRegMaskPair VMP(VReg, mask);
                      expectedDistances[VMP] = distance;
                    }
                  }
                }
              }
              break;
            }
          }
        }
      }
    }

    return expectedDistances;
  }

  void SetUp() override {
    // Only enable debug output if environment variable is set
    const char *debugEnv = std::getenv("AMDGPU_NUA_DEBUG");
    if (debugEnv && std::string(debugEnv) == "1") {
      DebugFlag = true;
      setCurrentDebugType("amdgpu-next-use");
    }
    Ctx = std::make_unique<LLVMContext>();
    TM = createAMDGPUTargetMachine("amdgcn-amd-", "gfx1200", "");
    if (!TM) {
      GTEST_SKIP() << "AMDGPU target not available";
    }
    static bool InitializedOnce = false;
    if (!InitializedOnce) {
      // Initialize required passes
      PassRegistry &PR = *PassRegistry::getPassRegistry();
      initializeMachineModuleInfoWrapperPassPass(PR);
      initializeMachineDominatorTreeWrapperPassPass(PR);
      initializeSlotIndexesWrapperPassPass(PR);
      initializeMachineLoopInfoWrapperPassPass(PR);
      InitializedOnce = true;
    }
  }

  // Helper to find all .mir files in a directory
  std::vector<std::string> findMirFiles(const std::string& dirPath) {
    std::vector<std::string> mirFiles;
    
    if (!std::filesystem::exists(dirPath)) {
      return mirFiles;
    }
    
    for (const auto& entry : std::filesystem::directory_iterator(dirPath)) {
      if (entry.is_regular_file() && entry.path().extension() == ".mir") {
        mirFiles.push_back(entry.path().filename().string());
      }
    }
    
    std::sort(mirFiles.begin(), mirFiles.end());
    return mirFiles;
  }
  
  // Helper to parse CHECK patterns from file comments
  std::vector<std::string> parseCheckPatterns(const std::string& filePath) {
    std::vector<std::string> patterns;
    std::ifstream file(filePath);
    std::string line;
    
    while (std::getline(file, line)) {
      if (line.find("# CHECK") == 0) {
        patterns.push_back(line);
      }
    }
    
    return patterns;
  }

  std::unique_ptr<Module> parseMIRFile(const std::string &FilePath,
                                       LLVMContext &Ctx,
                                       const TargetMachine &TM,
                                       legacy::PassManager &PM) {
    // 1) Add MMI wrapper first, get a handle to its MMI
    auto *MMIWP = new MachineModuleInfoWrapperPass(&TM);
    PM.add(MMIWP);
    MMI = &MMIWP->getMMI();

    // 2) Parse MIR
    ErrorOr<std::unique_ptr<MemoryBuffer>> FileBuffer = MemoryBuffer::getFile(FilePath);
    if (!FileBuffer) {
      return nullptr;
    }
    
    SMDiagnostic Err;
    auto MIRParser = createMIRParser(std::move(FileBuffer.get()), Ctx);
    if (!MIRParser) {
      return nullptr;
    }
    
    auto M = MIRParser->parseIRModule();
    if (!M) {
      return nullptr;
    }

    M->setTargetTriple(TM.getTargetTriple());
    M->setDataLayout(TM.createDataLayout());
    MMIWP->doInitialization(*M);

    if (MIRParser->parseMachineFunctions(*M, *MMI))
      return nullptr;
    return M;
  }

  NextUseResult &runNextUseAnalysis(MachineFunction &MF,
                                    legacy::PassManager &PM) {
    // Add our analysis pass at pre-emit stage (after most optimizations)
    PM.add(new NextUseAnalysisTestWrapper());

    PM.run(const_cast<Module &>(*MMI->getModule()));

    // Get the analysis result from the wrapper that was run
    return *NextUseAnalysisTestWrapper::Captured;
  }
};

// Parameterized test for all .mir files
class NextUseAnalysisParameterizedTest : public NextUseAnalysisTestBase, 
                                         public testing::WithParamInterface<std::string> {
};

std::string getTestDirectory() {
  // First try environment variable
  const char *testDirEnv = std::getenv("AMDGPU_NUA_TEST_DIR");
  if (testDirEnv) {
    return std::string(testDirEnv);
  }

  // Try to find relative to unit test binary
  // Unit tests are typically in build/unittests/Target/AMDGPU/
  // Source tests are in llvm/test/CodeGen/AMDGPU/NextUseAnalysis/
  std::filesystem::path currentPath = std::filesystem::current_path();

  // Look for the source tree from build directory
  std::vector<std::string> possiblePaths = {
      "../../../llvm/test/CodeGen/AMDGPU/NextUseAnalysis",
      "../../../../llvm/test/CodeGen/AMDGPU/NextUseAnalysis",
      "../../../../../llvm/test/CodeGen/AMDGPU/NextUseAnalysis"};

  for (const auto &path : possiblePaths) {
    std::filesystem::path testPath = currentPath / path;
    if (std::filesystem::exists(testPath)) {
      return testPath.string();
    }
  }

  return ""; // Not found
}

TEST_P(NextUseAnalysisParameterizedTest, ProcessMirFile) {
  std::string mirFileName = GetParam();
  
  // Get test directory from environment or use default
  std::string testDir = getTestDirectory();

  if (testDir.empty()) {
    GTEST_SKIP()
        << "NextUseAnalysis test directory not found. "
        << "Set AMDGPU_NUA_TEST_DIR environment variable or ensure "
        << "tests are run from build directory with source tree available.";
  }

  std::string fullPath = testDir + "/" + mirFileName;
  
  // Parse CHECK patterns from the file
  auto checkPatterns = parseCheckPatterns(fullPath);
  ASSERT_FALSE(checkPatterns.empty()) << "No CHECK patterns found in " << mirFileName;
  
  LLVMContext Ctx;
  legacy::PassManager PM;
  auto Module = parseMIRFile(fullPath, Ctx, *TM, PM);
  ASSERT_TRUE(Module) << "Failed to parse MIR file: " << mirFileName;

  for (auto &F : Module->functions()) {
    MachineFunction *MF = MMI->getMachineFunction(F);
    ASSERT_TRUE(MF) << "MachineFunction not found";

    // Initialize TRI and MRI from first MachineFunction if not already done
    if (!TRI) {
      TRI = MF->getSubtarget<GCNSubtarget>().getRegisterInfo();
      MRI = &MF->getRegInfo();
    }

    // Run NextUseAnalysis
    NextUseResult &NU = runNextUseAnalysis(*MF, PM);

    for (auto &MBB : *MF) {
      for (auto &MI : MBB) {
        // Convert MachineInstr to string for pattern matching
        std::string instrString = machineInstrToString(MI);

        // Parse expected distances from CHECK patterns for this instruction
        auto expectedDistances =
            parseExpectedDistances(checkPatterns, instrString, TRI, MRI);

        // Validate each expected distance
        for (const auto &expected : expectedDistances) {
          VRegMaskPair VMP = expected.first;
          unsigned expectedDistance = expected.second;

          unsigned actualDistance =
              NU.getNextUseDistance(MI.getIterator(), VMP);

          EXPECT_EQ(actualDistance, expectedDistance)
              << "Distance mismatch for register "
              << printReg(VMP.getVReg(), TRI,
                          getSubRegIndexForLaneMask(VMP.getLaneMask(), TRI),
                          MRI)
              << " in instruction: " << instrString.substr(0, 50) << "..."
              << " Expected: " << expectedDistance
              << " Actual: " << actualDistance;
        }
      }
    }
  }
  
}

// Generate test parameters from available .mir files
std::vector<std::string> getMirFiles() {
  const char *testFileEnv = std::getenv("AMDGPU_NUA_TEST_FILE");

  // If specific file is requested, test only that
  if (testFileEnv) {
    return {std::string(testFileEnv)};
  }

  // Use the same directory resolution as the test
  std::string testDir = getTestDirectory();

  // If no test directory found, return empty vector
  if (testDir.empty()) {
    return {};
  }

  // Find all .mir files in the directory
  std::vector<std::string> mirFiles;

  for (const auto &entry : std::filesystem::directory_iterator(testDir)) {
    if (entry.is_regular_file() && entry.path().extension() == ".mir") {
      mirFiles.push_back(entry.path().filename().string());
    }
  }

  std::sort(mirFiles.begin(), mirFiles.end());
  return mirFiles;
}

INSTANTIATE_TEST_SUITE_P(
  AllMirFiles,
  NextUseAnalysisParameterizedTest,
  testing::ValuesIn(getMirFiles()),
  [](const testing::TestParamInfo<std::string>& info) {
    std::string name = info.param;
    // Replace non-alphanumeric characters with underscores for valid test names
    std::replace_if(name.begin(), name.end(), [](char c) { 
      return !std::isalnum(c); 
    }, '_');
    return name;
  }
);

} // end anonymous namespace
