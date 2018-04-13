//===------------------ llvm-opt-report/OptReport.cpp ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file implements a tool that can parse the YAML optimization
/// records and generate an optimization summary annotated source listing
/// report.
///
//===----------------------------------------------------------------------===//

#include "llvm/Demangle/Demangle.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <map>
#include <set>

using namespace llvm;
using namespace llvm::yaml;

static cl::opt<bool> Help("h", cl::desc("Alias for -help"), cl::Hidden);

// Mark all our options with this category, everything else (except for -version
// and -help) will be hidden.
static cl::OptionCategory
    OptReportCategory("llvm-opt-report options");

static cl::opt<std::string>
  InputFileName(cl::Positional, cl::desc("<input>"), cl::init("-"),
                cl::cat(OptReportCategory));

static cl::opt<std::string>
  OutputFileName("o", cl::desc("Output file"), cl::init("-"),
                 cl::cat(OptReportCategory));

static cl::opt<std::string>
  InputRelDir("r", cl::desc("Root for relative input paths"), cl::init(""),
              cl::cat(OptReportCategory));

static cl::opt<bool>
  Succinct("s", cl::desc("Don't include vectorization factors, etc."),
           cl::init(false), cl::cat(OptReportCategory));

static cl::opt<bool>
  NoDemangle("no-demangle", cl::desc("Don't demangle function names"),
             cl::init(false), cl::cat(OptReportCategory));

namespace {
// For each location in the source file, the common per-transformation state
// collected.
struct OptReportLocationItemInfo {
  bool Analyzed = false;
  bool Transformed = false;

  OptReportLocationItemInfo &operator |= (
    const OptReportLocationItemInfo &RHS) {
    Analyzed |= RHS.Analyzed;
    Transformed |= RHS.Transformed;

    return *this;
  }

  bool operator < (const OptReportLocationItemInfo &RHS) const {
    if (Analyzed < RHS.Analyzed)
      return true;
    else if (Analyzed > RHS.Analyzed)
      return false;
    else if (Transformed < RHS.Transformed)
      return true;
    return false;
  }
};

// The per-location information collected for producing an optimization report.
struct OptReportLocationInfo {
  OptReportLocationItemInfo Inlined;
  OptReportLocationItemInfo Unrolled;
  OptReportLocationItemInfo Vectorized;

  int VectorizationFactor = 1;
  int InterleaveCount = 1;
  int UnrollCount = 1;

  OptReportLocationInfo &operator |= (const OptReportLocationInfo &RHS) {
    Inlined |= RHS.Inlined;
    Unrolled |= RHS.Unrolled;
    Vectorized |= RHS.Vectorized;

    VectorizationFactor =
      std::max(VectorizationFactor, RHS.VectorizationFactor);
    InterleaveCount = std::max(InterleaveCount, RHS.InterleaveCount);
    UnrollCount = std::max(UnrollCount, RHS.UnrollCount);

    return *this;
  }

  bool operator < (const OptReportLocationInfo &RHS) const {
    if (Inlined < RHS.Inlined)
      return true;
    else if (RHS.Inlined < Inlined)
      return false;
    else if (Unrolled < RHS.Unrolled)
      return true;
    else if (RHS.Unrolled < Unrolled)
      return false;
    else if (Vectorized < RHS.Vectorized)
      return true;
    else if (RHS.Vectorized < Vectorized || Succinct)
      return false;
    else if (VectorizationFactor < RHS.VectorizationFactor)
      return true;
    else if (VectorizationFactor > RHS.VectorizationFactor)
      return false;
    else if (InterleaveCount < RHS.InterleaveCount)
      return true;
    else if (InterleaveCount > RHS.InterleaveCount)
      return false;
    else if (UnrollCount < RHS.UnrollCount)
      return true;
    return false;
  }
};

typedef std::map<std::string, std::map<int, std::map<std::string, std::map<int,
          OptReportLocationInfo>>>> LocationInfoTy;
} // anonymous namespace

static void collectLocationInfo(yaml::Stream &Stream,
                                LocationInfoTy &LocationInfo) {
  SmallVector<char, 8> Tmp;

  // Note: We're using the YAML parser here directly, instead of using the
  // YAMLTraits implementation, because the YAMLTraits implementation does not
  // support a way to handle only a subset of the input keys (it will error out
  // if there is an input key that you don't map to your class), and
  // furthermore, it does not provide a way to handle the Args sequence of
  // key/value pairs, where the order must be captured and the 'String' key
  // might be repeated.
  for (auto &Doc : Stream) {
    auto *Root = dyn_cast<yaml::MappingNode>(Doc.getRoot());
    if (!Root)
      continue;

    bool Transformed = Root->getRawTag() == "!Passed";
    std::string Pass, File, Function;
    int Line = 0, Column = 1;

    int VectorizationFactor = 1;
    int InterleaveCount = 1;
    int UnrollCount = 1;

    for (auto &RootChild : *Root) {
      auto *Key = dyn_cast<yaml::ScalarNode>(RootChild.getKey());
      if (!Key)
        continue;
      StringRef KeyName = Key->getValue(Tmp);
      if (KeyName == "Pass") {
        auto *Value = dyn_cast<yaml::ScalarNode>(RootChild.getValue());
        if (!Value)
          continue;
        Pass = Value->getValue(Tmp);
      } else if (KeyName == "Function") {
        auto *Value = dyn_cast<yaml::ScalarNode>(RootChild.getValue());
        if (!Value)
          continue;
        Function = Value->getValue(Tmp);
      } else if (KeyName == "DebugLoc") {
        auto *DebugLoc = dyn_cast<yaml::MappingNode>(RootChild.getValue());
        if (!DebugLoc)
          continue;

        for (auto &DLChild : *DebugLoc) {
          auto *DLKey = dyn_cast<yaml::ScalarNode>(DLChild.getKey());
          if (!DLKey)
            continue;
          StringRef DLKeyName = DLKey->getValue(Tmp);
          if (DLKeyName == "File") {
            auto *Value = dyn_cast<yaml::ScalarNode>(DLChild.getValue());
            if (!Value)
              continue;
            File = Value->getValue(Tmp);
          } else if (DLKeyName == "Line") {
            auto *Value = dyn_cast<yaml::ScalarNode>(DLChild.getValue());
            if (!Value)
              continue;
            Value->getValue(Tmp).getAsInteger(10, Line);
          } else if (DLKeyName == "Column") {
            auto *Value = dyn_cast<yaml::ScalarNode>(DLChild.getValue());
            if (!Value)
              continue;
            Value->getValue(Tmp).getAsInteger(10, Column);
          }
        }
      } else if (KeyName == "Args") {
        auto *Args = dyn_cast<yaml::SequenceNode>(RootChild.getValue());
        if (!Args)
          continue;
        for (auto &ArgChild : *Args) {
          auto *ArgMap = dyn_cast<yaml::MappingNode>(&ArgChild);
          if (!ArgMap)
            continue;
          for (auto &ArgKV : *ArgMap) {
            auto *ArgKey = dyn_cast<yaml::ScalarNode>(ArgKV.getKey());
            if (!ArgKey)
              continue;
            StringRef ArgKeyName = ArgKey->getValue(Tmp);
            if (ArgKeyName == "VectorizationFactor") {
              auto *Value = dyn_cast<yaml::ScalarNode>(ArgKV.getValue());
              if (!Value)
                continue;
              Value->getValue(Tmp).getAsInteger(10, VectorizationFactor);
            } else if (ArgKeyName == "InterleaveCount") {
              auto *Value = dyn_cast<yaml::ScalarNode>(ArgKV.getValue());
              if (!Value)
                continue;
              Value->getValue(Tmp).getAsInteger(10, InterleaveCount);
            } else if (ArgKeyName == "UnrollCount") {
              auto *Value = dyn_cast<yaml::ScalarNode>(ArgKV.getValue());
              if (!Value)
                continue;
              Value->getValue(Tmp).getAsInteger(10, UnrollCount);
            }
          }
        }
      }
    }

    if (Line < 1 || File.empty())
      continue;

    // We track information on both actual and potential transformations. This
    // way, if there are multiple possible things on a line that are, or could
    // have been transformed, we can indicate that explicitly in the output.
    auto UpdateLLII = [Transformed](OptReportLocationItemInfo &LLII) {
      LLII.Analyzed = true;
      if (Transformed)
        LLII.Transformed = true;
    };

    if (Pass == "inline") {
      auto &LI = LocationInfo[File][Line][Function][Column];
      UpdateLLII(LI.Inlined);
    } else if (Pass == "loop-unroll") {
      auto &LI = LocationInfo[File][Line][Function][Column];
      LI.UnrollCount = UnrollCount;
      UpdateLLII(LI.Unrolled);
    } else if (Pass == "loop-vectorize") {
      auto &LI = LocationInfo[File][Line][Function][Column];
      LI.VectorizationFactor = VectorizationFactor;
      LI.InterleaveCount = InterleaveCount;
      UpdateLLII(LI.Vectorized);
    }
  }
}

static bool readLocationInfo(LocationInfoTy &LocationInfo) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> Buf =
      MemoryBuffer::getFileOrSTDIN(InputFileName);
  if (std::error_code EC = Buf.getError()) {
    errs() << "error: Can't open file " << InputFileName << ": " <<
              EC.message() << "\n";
    return false;
  }

  SourceMgr SM;
  yaml::Stream Stream(Buf.get()->getBuffer(), SM);
  collectLocationInfo(Stream, LocationInfo);

  return true; 
}

static bool writeReport(LocationInfoTy &LocationInfo) {
  std::error_code EC;
  llvm::raw_fd_ostream OS(OutputFileName, EC,
              llvm::sys::fs::F_Text);
  if (EC) {
    errs() << "error: Can't open file " << OutputFileName << ": " <<
              EC.message() << "\n";
    return false;
  }

  bool FirstFile = true;
  for (auto &FI : LocationInfo) {
    SmallString<128> FileName(FI.first);
    if (!InputRelDir.empty()) {
      if (std::error_code EC = sys::fs::make_absolute(InputRelDir, FileName)) {
        errs() << "error: Can't resolve file path to " << FileName << ": " <<
                  EC.message() << "\n";
        return false;
      }
    }

    const auto &FileInfo = FI.second;

    ErrorOr<std::unique_ptr<MemoryBuffer>> Buf =
        MemoryBuffer::getFile(FileName);
    if (std::error_code EC = Buf.getError()) {
      errs() << "error: Can't open file " << FileName << ": " <<
                EC.message() << "\n";
      return false;
    }

    if (FirstFile)
      FirstFile = false;
    else
      OS << "\n";

    OS << "< " << FileName << "\n";

    // Figure out how many characters we need for the vectorization factors
    // and similar.
    OptReportLocationInfo MaxLI;
    for (auto &FLI : FileInfo)
      for (auto &FI : FLI.second)
        for (auto &LI : FI.second)
          MaxLI |= LI.second;

    bool NothingInlined = !MaxLI.Inlined.Transformed;
    bool NothingUnrolled = !MaxLI.Unrolled.Transformed;
    bool NothingVectorized = !MaxLI.Vectorized.Transformed;

    unsigned VFDigits = llvm::utostr(MaxLI.VectorizationFactor).size();
    unsigned ICDigits = llvm::utostr(MaxLI.InterleaveCount).size();
    unsigned UCDigits = llvm::utostr(MaxLI.UnrollCount).size();

    // Figure out how many characters we need for the line numbers.
    int64_t NumLines = 0;
    for (line_iterator LI(*Buf.get(), false); LI != line_iterator(); ++LI)
      ++NumLines;

    unsigned LNDigits = llvm::utostr(NumLines).size();

    for (line_iterator LI(*Buf.get(), false); LI != line_iterator(); ++LI) {
      int64_t L = LI.line_number();
      auto LII = FileInfo.find(L);

      auto PrintLine = [&](bool PrintFuncName,
                           const std::set<std::string> &FuncNameSet) {
        OptReportLocationInfo LLI;

        std::map<int, OptReportLocationInfo> ColsInfo;
        unsigned InlinedCols = 0, UnrolledCols = 0, VectorizedCols = 0;

        if (LII != FileInfo.end() && !FuncNameSet.empty()) {
          const auto &LineInfo = LII->second;

          for (auto &CI : LineInfo.find(*FuncNameSet.begin())->second) {
            int Col = CI.first;
            ColsInfo[Col] = CI.second;
            InlinedCols += CI.second.Inlined.Analyzed;
            UnrolledCols += CI.second.Unrolled.Analyzed;
            VectorizedCols += CI.second.Vectorized.Analyzed;
            LLI |= CI.second;
          }
        }

        if (PrintFuncName) {
          OS << "  > ";

          bool FirstFunc = true;
          for (const auto &FuncName : FuncNameSet) {
            if (FirstFunc)
              FirstFunc = false;
            else
              OS << ", ";

            bool Printed = false;
            if (!NoDemangle) {
              int Status = 0;
              char *Demangled =
                itaniumDemangle(FuncName.c_str(), nullptr, nullptr, &Status);
              if (Demangled && Status == 0) {
                OS << Demangled;
                Printed = true;
              }

              if (Demangled)
                std::free(Demangled);
            }

            if (!Printed)
              OS << FuncName;
          } 

          OS << ":\n";
        }

        // We try to keep the output as concise as possible. If only one thing on
        // a given line could have been inlined, vectorized, etc. then we can put
        // the marker on the source line itself. If there are multiple options
        // then we want to distinguish them by placing the marker for each
        // transformation on a separate line following the source line. When we
        // do this, we use a '^' character to point to the appropriate column in
        // the source line.

        std::string USpaces(Succinct ? 0 : UCDigits, ' ');
        std::string VSpaces(Succinct ? 0 : VFDigits + ICDigits + 1, ' ');

        auto UStr = [UCDigits](OptReportLocationInfo &LLI) {
          std::string R;
          raw_string_ostream RS(R);

          if (!Succinct) {
            RS << LLI.UnrollCount;
            RS << std::string(UCDigits - RS.str().size(), ' ');
          }

          return RS.str();
        };

        auto VStr = [VFDigits,
                     ICDigits](OptReportLocationInfo &LLI) -> std::string {
          std::string R;
          raw_string_ostream RS(R);

          if (!Succinct) {
            RS << LLI.VectorizationFactor << "," << LLI.InterleaveCount;
            RS << std::string(VFDigits + ICDigits + 1 - RS.str().size(), ' ');
          }

          return RS.str();
        };

        OS << llvm::format_decimal(L, LNDigits) << " ";
        OS << (LLI.Inlined.Transformed && InlinedCols < 2 ? "I" :
                (NothingInlined ? "" : " "));
        OS << (LLI.Unrolled.Transformed && UnrolledCols < 2 ?
                "U" + UStr(LLI) : (NothingUnrolled ? "" : " " + USpaces));
        OS << (LLI.Vectorized.Transformed && VectorizedCols < 2 ?
                "V" + VStr(LLI) : (NothingVectorized ? "" : " " + VSpaces));

        OS << " | " << *LI << "\n";

        for (auto &J : ColsInfo) {
          if ((J.second.Inlined.Transformed && InlinedCols > 1) ||
              (J.second.Unrolled.Transformed && UnrolledCols > 1) ||
              (J.second.Vectorized.Transformed && VectorizedCols > 1)) {
            OS << std::string(LNDigits + 1, ' ');
            OS << (J.second.Inlined.Transformed &&
                   InlinedCols > 1 ? "I" : (NothingInlined ? "" : " "));
            OS << (J.second.Unrolled.Transformed &&
                   UnrolledCols > 1 ? "U" + UStr(J.second) :
                     (NothingUnrolled ? "" : " " + USpaces));
            OS << (J.second.Vectorized.Transformed &&
                   VectorizedCols > 1 ? "V" + VStr(J.second) :
                     (NothingVectorized ? "" : " " + VSpaces));

            OS << " | " << std::string(J.first - 1, ' ') << "^\n";
          }
        }
      };

      // We need to figure out if the optimizations for this line were the same
      // in each function context. If not, then we want to group the similar
      // function contexts together and display each group separately. If
      // they're all the same, then we only display the line once without any
      // additional markings.
      std::map<std::map<int, OptReportLocationInfo>,
               std::set<std::string>> UniqueLIs;

      OptReportLocationInfo AllLI;
      if (LII != FileInfo.end()) {
        const auto &FuncLineInfo = LII->second;
        for (const auto &FLII : FuncLineInfo) {
          UniqueLIs[FLII.second].insert(FLII.first);

          for (const auto &OI : FLII.second)
            AllLI |= OI.second;
        }
      }

      bool NothingHappened = !AllLI.Inlined.Transformed &&
                             !AllLI.Unrolled.Transformed &&
                             !AllLI.Vectorized.Transformed;
      if (UniqueLIs.size() > 1 && !NothingHappened) {
        OS << " [[\n";
        for (const auto &FSLI : UniqueLIs)
          PrintLine(true, FSLI.second);
        OS << " ]]\n";
      } else if (UniqueLIs.size() == 1) {
        PrintLine(false, UniqueLIs.begin()->second);
      } else {
        PrintLine(false, std::set<std::string>());
      }
    }
  }

  return true;
}

int main(int argc, const char **argv) {
  InitLLVM X(argc, argv);

  cl::HideUnrelatedOptions(OptReportCategory);
  cl::ParseCommandLineOptions(
      argc, argv,
      "A tool to generate an optimization report from YAML optimization"
      " record files.\n");

  if (Help) {
    cl::PrintHelpMessage();
    return 0;
  }

  LocationInfoTy LocationInfo;
  if (!readLocationInfo(LocationInfo))
    return 1;
  if (!writeReport(LocationInfo))
    return 1; 

  return 0;
}

