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

#include "AMDGPUNextUseAnalysis.h"

using namespace llvm;

namespace {

// Helper wrapper to store analysis results for unit testing.
// NOTE: Static result storage is intentional for unit tests.
// The legacy PassManager doesn't expose the pass instance after run(),
// so we capture the result here. Unit tests run sequentially in gtest,
// so this doesn't cause isolation issues.
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

// ============================================================================
// String Parsing Utilities
// ============================================================================
// These functions replace std::regex for ~100-1000x faster pattern matching.
// StringRef operations are direct pointer arithmetic with no regex overhead.

/// Parse "CHECK: Vreg: %N:subreg[ D ]" or "CHECK: Vreg: %N[ D ]" pattern.
/// Returns true if matched, fills RegNum, SubRegName, Distance.
bool parseVregPattern(StringRef Line, unsigned &RegNum, StringRef &SubRegName,
                      unsigned &Distance) {
  if (!Line.consume_front("CHECK:"))
    return false;
  Line = Line.ltrim();

  if (!Line.consume_front("Vreg:"))
    return false;
  Line = Line.ltrim();

  if (!Line.consume_front("%"))
    return false;

  // Parse register number
  size_t NumLen = Line.find_first_not_of("0123456789");
  if (NumLen == 0)
    return false;
  if (Line.substr(0, NumLen).getAsInteger(10, RegNum))
    return false;
  Line = Line.drop_front(NumLen);

  // Optional sub-register: ":subreg_name"
  SubRegName = StringRef();
  if (Line.consume_front(":")) {
    size_t SubEnd = Line.find('[');
    if (SubEnd == StringRef::npos)
      return false;
    SubRegName = Line.substr(0, SubEnd);
    Line = Line.drop_front(SubEnd);
  }

  // Parse "[ D ]"
  if (!Line.consume_front("["))
    return false;
  Line = Line.ltrim();

  size_t DistLen = Line.find_first_not_of("0123456789");
  if (DistLen == 0)
    return false;
  return !Line.substr(0, DistLen).getAsInteger(10, Distance);
}

class NextUseAnalysisTestBase : public testing::Test {
protected:
  std::unique_ptr<LLVMContext> Ctx;
  std::unique_ptr<Module> M;
  std::unique_ptr<const GCNTargetMachine> TM;
  MachineModuleInfo *MMI = nullptr;

  // Add TRI and MRI as class members - initialized from first MachineFunction
  const SIRegisterInfo *TRI = nullptr;
  const MachineRegisterInfo *MRI = nullptr;

  // Helper to print MachineInstr into a reusable buffer (avoids allocation)
  void printMachineInstr(const MachineInstr &MI, SmallVectorImpl<char> &Buf) {
    Buf.clear();
    raw_svector_ostream OS(Buf);
    MI.print(OS, /*IsStandalone=*/false, /*SkipOpers=*/false,
             /*SkipDebugLoc=*/true, /*AddNewLine=*/false);
  }

  // Helper function to map sub-register names to LaneBitmask values using LLVM
  // API. SubRegName may be empty for full register.
  LaneBitmask getSubRegLaneMask(StringRef SubRegName, Register VReg,
                                const SIRegisterInfo *TRI,
                                const MachineRegisterInfo *MRI) {
    if (SubRegName.empty()) {
      return MRI->getMaxLaneMaskForVReg(VReg); // Proper full register mask
    }

    for (unsigned I = 1, E = TRI->getNumSubRegIndices(); I < E; ++I) {
      const char *Name = TRI->getSubRegIndexName(I);
      if (Name && SubRegName == Name) {
        return TRI->getSubRegIndexLaneMask(I);
      }
    }

    // If unknown sub-register, default to full register
    return MRI->getMaxLaneMaskForVReg(VReg);
  }

  // Parse expected distances from CHECK patterns for a specific instruction.
  // Uses StringRef-based parsing for performance (avoids std::regex overhead).
  llvm::DenseMap<VRegMaskPair, unsigned>
  parseExpectedDistances(const std::vector<std::string> &CheckPatterns,
                         StringRef InstrRef, const SIRegisterInfo *TRI,
                         const MachineRegisterInfo *MRI) {
    llvm::DenseMap<VRegMaskPair, unsigned> ExpectedDistances;

    // Trim the instruction (handles whitespace and \n at ends)
    InstrRef = InstrRef.trim();

    // Extract instruction core (everything after first space)
    size_t SpacePos = InstrRef.find(' ');
    StringRef InstrCore = SpacePos == StringRef::npos
                              ? InstrRef
                              : InstrRef.drop_front(SpacePos + 1);

    // Find matching CHECK pattern index
    size_t MatchIdx = StringRef::npos;
    for (size_t I = 0; I < CheckPatterns.size(); ++I) {
      StringRef Pattern = CheckPatterns[I];

      size_t InstrPos = Pattern.find("CHECK: Instr:");
      if (InstrPos == StringRef::npos)
        continue;

      // Extract and trim the CHECK instruction
      StringRef CheckInstr = Pattern.drop_front(InstrPos + 13).trim();

      // Extract CHECK instruction core
      size_t CheckSpacePos = CheckInstr.find(' ');
      StringRef CheckCore = CheckSpacePos == StringRef::npos
                                ? CheckInstr
                                : CheckInstr.drop_front(CheckSpacePos + 1);

      if (CheckCore != InstrCore)
        continue;

      // Verify destination register matches (prefix up to '=')
      size_t CheckEq = CheckInstr.find('=');
      size_t InstrEq = InstrRef.find('=');
      if (CheckEq != StringRef::npos && InstrEq != StringRef::npos) {
        StringRef CheckDest = CheckInstr.substr(0, CheckEq).trim();
        StringRef InstrDest = InstrRef.substr(0, InstrEq).trim();
        if (CheckDest == InstrDest) {
          MatchIdx = I;
          break;
        }
      }
    }

    if (MatchIdx == StringRef::npos)
      return ExpectedDistances;

    // Collect Vreg patterns following the matched instruction
    llvm::DenseMap<Register, SmallVector<std::pair<LaneBitmask, unsigned>, 4>>
        AllMaskDistances;

    for (size_t J = MatchIdx + 1; J < CheckPatterns.size(); ++J) {
      StringRef DistPattern = CheckPatterns[J];

      // Stop at next instruction or section marker
      if (DistPattern.contains("CHECK: Instr:") ||
          DistPattern.contains("CHECK: ---") ||
          DistPattern.contains("CHECK-LABEL:") ||
          DistPattern.contains("Block End Distances:")) {
        break;
      }

      unsigned RegNum, Distance;
      StringRef SubRegName;
      if (parseVregPattern(DistPattern, RegNum, SubRegName, Distance)) {
        Register VReg = Register::index2VirtReg(RegNum);
        LaneBitmask Mask = getSubRegLaneMask(SubRegName, VReg, TRI, MRI);
        AllMaskDistances[VReg].push_back({Mask, Distance});
      }
    }

    // Build ExpectedDistances using overlap semantics:
    // For each VMP, find minimum distance among all overlapping masks
    for (const auto &[VReg, MaskDistPairs] : AllMaskDistances) {
      for (const auto &[QueryMask, _] : MaskDistPairs) {
        unsigned MinDist = std::numeric_limits<unsigned>::max();
        for (const auto &[StoredMask, Dist] : MaskDistPairs) {
          if ((QueryMask & StoredMask).any())
            MinDist = std::min(MinDist, Dist);
        }
        ExpectedDistances[VRegMaskPair(VReg, QueryMask)] = MinDist;
      }
    }

    return ExpectedDistances;
  }

  void SetUp() override {
    // Only enable debug output if environment variable is set
    const char *DebugEnv = std::getenv("AMDGPU_NUA_DEBUG");
    if (DebugEnv && std::string(DebugEnv) == "1") {
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
  std::vector<std::string> findMirFiles(const std::string &DirPath) {
    std::vector<std::string> MirFiles;

    if (!std::filesystem::exists(DirPath)) {
      return MirFiles;
    }

    for (const auto &Entry : std::filesystem::directory_iterator(DirPath)) {
      if (Entry.is_regular_file() && Entry.path().extension() == ".mir") {
        MirFiles.push_back(Entry.path().filename().string());
      }
    }

    std::sort(MirFiles.begin(), MirFiles.end());
    return MirFiles;
  }

  // Helper to parse CHECK patterns from file comments
  std::vector<std::string> parseCheckPatterns(const std::string &FilePath) {
    std::vector<std::string> Patterns;
    std::ifstream File(FilePath);
    std::string Line;

    while (std::getline(File, Line)) {
      // Strip "# " prefix from MIR comment lines, store just the CHECK
      // directive
      if (Line.find("# CHECK") == 0 && Line.size() > 2) {
        Patterns.push_back(Line.substr(2));
      }
    }

    return Patterns;
  }

  std::unique_ptr<Module> parseMIRString(const std::string &MIRContent,
                                         LLVMContext &Ctx,
                                         const TargetMachine &TM,
                                         legacy::PassManager &PM) {
    // 1) Add MMI wrapper first, get a handle to its MMI
    auto *MMIWP = new MachineModuleInfoWrapperPass(&TM);
    PM.add(MMIWP);
    MMI = &MMIWP->getMMI();

    // 2) Parse MIR from string
    auto MemBuffer = MemoryBuffer::getMemBuffer(MIRContent, "inline_mir");

    SMDiagnostic Err;
    auto MIRParser = createMIRParser(std::move(MemBuffer), Ctx);
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

  std::unique_ptr<Module> parseMIRFile(const std::string &FilePath,
                                       LLVMContext &Ctx,
                                       const TargetMachine &TM,
                                       legacy::PassManager &PM) {
    // Read file content
    ErrorOr<std::unique_ptr<MemoryBuffer>> FileBuffer =
        MemoryBuffer::getFile(FilePath);
    if (!FileBuffer) {
      return nullptr;
    }

    // Reuse parseMIRString with file content
    return parseMIRString(FileBuffer.get()->getBuffer().str(), Ctx, TM, PM);
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
class NextUseAnalysisParameterizedTest
    : public NextUseAnalysisTestBase,
      public testing::WithParamInterface<std::string> {};

// Allow uninstantiated test when test files are not available
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(NextUseAnalysisParameterizedTest);

std::string getTestDirectory() {
  // Allow environment variable override
  const char *TestDirEnv = std::getenv("AMDGPU_NUA_TEST_DIR");
  if (TestDirEnv) {
    return std::string(TestDirEnv);
  }

  // Use CMake-defined path (set via LLVM_MAIN_SRC_DIR in CMakeLists.txt)
#ifdef AMDGPU_NUA_TEST_DIR
  return AMDGPU_NUA_TEST_DIR;
#else
  return "";
#endif
}

// Generate test parameters from available .mir files
std::vector<std::string> getMirFiles() {
  const char *TestFileEnv = std::getenv("AMDGPU_NUA_TEST_FILE");

  // If specific file is requested, test only that
  if (TestFileEnv) {
    return {std::string(TestFileEnv)};
  }

  // Use the same directory resolution as the test
  std::string TestDir = getTestDirectory();

  // If no test directory found, print help and return empty vector
  if (TestDir.empty()) {
    std::cerr << "Warning: NextUseAnalysis test directory not found.\n"
              << "Run from build/Debug (or build/Release) directory:\n"
              << "  cd build/Debug && ./unittests/Target/AMDGPU/AMDGPUTests "
              << "--gtest_filter=\"AllMirFiles*\"\n"
              << "Or set AMDGPU_NUA_TEST_DIR to the test directory:\n"
              << "  AMDGPU_NUA_TEST_DIR=/path/to/llvm/test/CodeGen/AMDGPU/"
              << "NextUseAnalysis\n";
    return {};
  }

  // Find all .mir files in the directory
  std::vector<std::string> MirFiles;

  for (const auto &Entry : std::filesystem::directory_iterator(TestDir)) {
    if (Entry.is_regular_file() && Entry.path().extension() == ".mir") {
      MirFiles.push_back(Entry.path().filename().string());
    }
  }

  std::sort(MirFiles.begin(), MirFiles.end());
  return MirFiles;
}

TEST_P(NextUseAnalysisParameterizedTest, ProcessMirFile) {
  std::string MirFileName = GetParam();

  // Get test directory from environment or use default
  std::string TestDir = getTestDirectory();

  if (TestDir.empty()) {
    GTEST_SKIP() << "NextUseAnalysis test directory not found.\n"
                 << "Run from build/Debug (or build/Release) directory:\n"
                 << "  cd build/Debug && ./unittests/Target/AMDGPU/AMDGPUTests "
                 << "--gtest_filter=\"AllMirFiles*\"\n"
                 << "Or set AMDGPU_NUA_TEST_DIR to the test directory:\n"
                 << "  "
                    "AMDGPU_NUA_TEST_DIR=/path/to/llvm/test/CodeGen/AMDGPU/"
                    "NextUseAnalysis";
  }

  std::string FullPath = TestDir + "/" + MirFileName;

  // Parse CHECK patterns from the file
  auto CheckPatterns = parseCheckPatterns(FullPath);
  ASSERT_FALSE(CheckPatterns.empty())
      << "No CHECK patterns found in " << MirFileName;

  LLVMContext Ctx;
  legacy::PassManager PM;
  auto Module = parseMIRFile(FullPath, Ctx, *TM, PM);
  ASSERT_TRUE(Module) << "Failed to parse MIR file: " << MirFileName;

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

    // Reusable buffer for instruction printing (avoids allocation per instr)
    SmallString<256> InstrBuf;

    for (auto &MBB : *MF) {
      for (auto &MI : MBB) {
        // Print MachineInstr into reusable buffer
        printMachineInstr(MI, InstrBuf);

        // Parse expected distances from CHECK patterns for this instruction
        auto ExpectedDistances =
            parseExpectedDistances(CheckPatterns, InstrBuf, TRI, MRI);

        // Validate each expected distance
        for (const auto &Expected : ExpectedDistances) {
          VRegMaskPair VMP = Expected.first;
          unsigned ExpectedDistance = Expected.second;

          unsigned ActualDistance =
              NU.getNextUseDistance(MI.getIterator(), VMP);

          EXPECT_EQ(ActualDistance, ExpectedDistance)
              << "Distance mismatch for register "
              << printReg(VMP.getVReg(), TRI,
                          TRI->getSubRegIndexForLaneMask(VMP.getLaneMask()),
                          MRI)
              << " in instruction: " << StringRef(InstrBuf).substr(0, 50)
              << "..."
              << " Expected: " << ExpectedDistance
              << " Actual: " << ActualDistance;
        }
      }
    }
  }
}

// Test getSortedSubregUses API with minimal SSA MIR pattern
TEST_F(NextUseAnalysisTestBase, GetSortedSubregUsesDistanceOrdering) {
  // Minimal MIR pattern based on real SSA spiller case:
  // Large register with sub-register accesses at different distances
  // sub0 is accessed last (furthest distance) and should appear first in sorted
  // result
  const char *MIR = R"MIR(
--- |
  target triple = "amdgcn"
  define void @getSortedSubregUses_test() { ret void }

---
name: getSortedSubregUses_test
body: |
  bb.0:
    ; Create large register with all 32 sub-registers
    %0:vreg_1024 = IMPLICIT_DEF

    ; Query point: %0 is now live and has upcoming sub-register uses
    %1:vgpr_32 = COPY $vgpr0      ; Query getSortedSubregUses here

    ; Multiple sub-register accesses in reverse order (sub31 -> sub0)
    ; This creates different next-use distances for each sub-register
    %10:vgpr_32 = COPY %0.sub31   ; Distance = 1 (closest)
    %11:vgpr_32 = COPY %0.sub30   ; Distance = 2
    %12:vgpr_32 = COPY %0.sub29   ; Distance = 3
    %13:vgpr_32 = COPY %0.sub28   ; Distance = 4

    ; Create REG_SEQUENCE with some of the copied values
    %20:vreg_128 = REG_SEQUENCE %13, %subreg.sub0, %12, %subreg.sub1, %11, %subreg.sub2, %10, %subreg.sub3

    ; Continue with more sub-register accesses
    %14:vgpr_32 = COPY %0.sub3    ; Distance = 6
    %15:vgpr_32 = COPY %0.sub2    ; Distance = 7
    %16:vgpr_32 = COPY %0.sub1    ; Distance = 8
    %17:vgpr_32 = COPY %0.sub0    ; Distance = 9 (furthest)

    ; Use the copied sub-registers
    %21:vreg_128 = REG_SEQUENCE %17, %subreg.sub0, %16, %subreg.sub1, %15, %subreg.sub2, %14, %subreg.sub3

    ; Store to make them live
    GLOBAL_STORE_DWORDX4 undef %30:vreg_64, %20, 0, 0, implicit $exec :: (store (s128), addrspace 1)
    GLOBAL_STORE_DWORDX4 undef %31:vreg_64, %21, 0, 0, implicit $exec :: (store (s128), addrspace 1)

...
)MIR";

  // Parse MIR from string using the new helper function
  LLVMContext Ctx;
  legacy::PassManager PM;
  auto Module = parseMIRString(MIR, Ctx, *TM, PM);
  ASSERT_TRUE(Module) << "Failed to parse MIR";

  // Get the MachineFunction
  auto &F = *Module->functions().begin();
  MachineFunction *MF = MMI->getMachineFunction(F);
  ASSERT_TRUE(MF) << "MachineFunction not found";

  // Initialize TRI and MRI if not already done
  if (!TRI) {
    TRI = MF->getSubtarget<GCNSubtarget>().getRegisterInfo();
    MRI = &MF->getRegInfo();
  }

  // Run NextUseAnalysis using the existing method
  NextUseResult &NU = runNextUseAnalysis(*MF, PM);

  // Find the COPY instruction (our query point where %0 is live but not yet
  // used)
  MachineBasicBlock &MBB = *MF->begin();
  auto QueryIt =
      std::find_if(MBB.begin(), MBB.end(), [](const MachineInstr &MI) {
        return MI.getOpcode() == TargetOpcode::COPY &&
               MI.getOperand(0).isReg() &&
               MI.getOperand(0).getReg() == Register::index2VirtReg(1); // %1
      });
  ASSERT_NE(QueryIt, MBB.end())
      << "Could not find COPY instruction for query point";

  // Get the virtual register number for %0 (defined by the previous
  // IMPLICIT_DEF)
  auto ImplicitDefIt =
      std::find_if(MBB.begin(), MBB.end(), [](const MachineInstr &MI) {
        return MI.getOpcode() == TargetOpcode::IMPLICIT_DEF;
      });
  ASSERT_NE(ImplicitDefIt, MBB.end())
      << "Could not find IMPLICIT_DEF instruction";

  Register VReg = ImplicitDefIt->getOperand(0).getReg();
  ASSERT_TRUE(VReg.isVirtual()) << "Expected virtual register";

  // Test getSortedSubregUses at the COPY instruction (after %0 is defined,
  // before it's used)
  LaneBitmask FullMask = MRI->getMaxLaneMaskForVReg(VReg);
  VRegMaskPair VMP(VReg, FullMask);

  SmallVector<VRegMaskPair> SortedUses = NU.getSortedSubregUses(QueryIt, VMP);

  // Verify that we got results
  ASSERT_FALSE(SortedUses.empty())
      << "getSortedSubregUses should return sub-register uses";

  // The key test: sub0 (accessed last at distance 9) should appear first
  // in the sorted result since getSortedSubregUses returns furthest uses first
  bool FoundSub0First = false;
  if (!SortedUses.empty()) {
    // Get the lane mask for sub0
    LaneBitmask Sub0Mask = TRI->getSubRegIndexLaneMask(AMDGPU::sub0);

    // Check if the first entry corresponds to sub0
    if (SortedUses[0].getLaneMask() == Sub0Mask) {
      FoundSub0First = true;
    }
  }

  EXPECT_TRUE(FoundSub0First) << "sub0 (furthest use) should appear first in "
                                 "getSortedSubregUses result";

  // Verify we have exactly 8 sub-register uses (sub0, sub1, sub2, sub3, sub28,
  // sub29, sub30, sub31)
  ASSERT_EQ(SortedUses.size(), 8u)
      << "Expected exactly 8 sub-register uses, got " << SortedUses.size();

  // Define expected sub-registers in order of decreasing distance (furthest
  // first) Based on our MIR: sub0 (dist 9), sub1 (dist 8), sub2 (dist 7), sub3
  // (dist 6),
  //                   sub28 (dist 4), sub29 (dist 3), sub30 (dist 2), sub31
  //                   (dist 1)
  std::vector<unsigned> ExpectedSubRegs = {
      AMDGPU::sub0,  // Distance 9 (furthest)
      AMDGPU::sub1,  // Distance 8
      AMDGPU::sub2,  // Distance 7
      AMDGPU::sub3,  // Distance 6
      AMDGPU::sub28, // Distance 4
      AMDGPU::sub29, // Distance 3
      AMDGPU::sub30, // Distance 2
      AMDGPU::sub31  // Distance 1 (closest)
  };

  // Verify exact order: furthest uses first
  for (size_t I = 0; I < ExpectedSubRegs.size(); ++I) {
    LaneBitmask ExpectedMask = TRI->getSubRegIndexLaneMask(ExpectedSubRegs[I]);
    LaneBitmask ActualMask = SortedUses[I].getLaneMask();

    EXPECT_EQ(ActualMask, ExpectedMask)
        << "Position " << I << ": Expected sub-register "
        << TRI->getSubRegIndexName(ExpectedSubRegs[I]) << " (mask "
        << ExpectedMask.getAsInteger() << "), "
        << "but got mask " << ActualMask.getAsInteger();
  }

  // Additional verification: all entries should reference the same VReg (%0)
  for (const auto &Use : SortedUses) {
    EXPECT_EQ(Use.getVReg(), VReg)
        << "All sorted uses should reference the same virtual register";
  }
}

INSTANTIATE_TEST_SUITE_P(AllMirFiles, NextUseAnalysisParameterizedTest,
                         testing::ValuesIn(getMirFiles()),
                         [](const testing::TestParamInfo<std::string> &info) {
                           std::string name = info.param;
                           // Replace non-alphanumeric characters with
                           // underscores for valid test names
                           std::replace_if(
                               name.begin(), name.end(),
                               [](char c) { return !std::isalnum(c); }, '_');
                           return name;
                         });

} // end anonymous namespace
