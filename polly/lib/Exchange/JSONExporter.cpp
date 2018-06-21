//===-- JSONExporter.cpp  - Export Scops as JSON  -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Export the Scops build by ScopInfo pass as a JSON file.
//
//===----------------------------------------------------------------------===//

#include "polly/JSONExporter.h"
#include "polly/DependenceInfo.h"
#include "polly/LinkAllPasses.h"
#include "polly/Options.h"
#include "polly/ScopInfo.h"
#include "polly/ScopPass.h"
#include "polly/Support/ScopLocation.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/RegionInfo.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "isl/constraint.h"
#include "isl/map.h"
#include "isl/printer.h"
#include "isl/set.h"
#include "isl/union_map.h"
#include "json/reader.h"
#include "json/writer.h"
#include <memory>
#include <string>
#include <system_error>

using namespace llvm;
using namespace polly;

#define DEBUG_TYPE "polly-import-jscop"

STATISTIC(NewAccessMapFound, "Number of updated access functions");

namespace {
static cl::opt<std::string>
    ImportDir("polly-import-jscop-dir",
              cl::desc("The directory to import the .jscop files from."),
              cl::Hidden, cl::value_desc("Directory path"), cl::ValueRequired,
              cl::init("."), cl::cat(PollyCategory));

static cl::opt<std::string>
    ImportPostfix("polly-import-jscop-postfix",
                  cl::desc("Postfix to append to the import .jsop files."),
                  cl::Hidden, cl::value_desc("File postfix"), cl::ValueRequired,
                  cl::init(""), cl::cat(PollyCategory));

struct JSONExporter : public ScopPass {
  static char ID;
  explicit JSONExporter() : ScopPass(ID) {}

  /// Export the SCoP @p S to a JSON file.
  bool runOnScop(Scop &S) override;

  /// Print the SCoP @p S as it is exported.
  void printScop(raw_ostream &OS, Scop &S) const override;

  /// Register all analyses and transformation required.
  void getAnalysisUsage(AnalysisUsage &AU) const override;
};

struct JSONImporter : public ScopPass {
  static char ID;
  std::vector<std::string> NewAccessStrings;
  explicit JSONImporter() : ScopPass(ID) {}
  /// Import new access functions for SCoP @p S from a JSON file.
  bool runOnScop(Scop &S) override;

  /// Print the SCoP @p S and the imported access functions.
  void printScop(raw_ostream &OS, Scop &S) const override;

  /// Register all analyses and transformation required.
  void getAnalysisUsage(AnalysisUsage &AU) const override;
};
} // namespace

static std::string getFileName(Scop &S, StringRef Suffix = "") {
  std::string FunctionName = S.getFunction().getName();
  std::string FileName = FunctionName + "___" + S.getNameStr() + ".jscop";

  if (Suffix != "")
    FileName += "." + Suffix.str();

  return FileName;
}

/// Export all arrays from the Scop.
///
/// @param S The Scop containing the arrays.
///
/// @returns Json::Value containing the arrays.
static Json::Value exportArrays(const Scop &S) {
  Json::Value Arrays;
  std::string Buffer;
  llvm::raw_string_ostream RawStringOstream(Buffer);

  for (auto &SAI : S.arrays()) {
    if (!SAI->isArrayKind())
      continue;

    Json::Value Array;
    Array["name"] = SAI->getName();
    unsigned i = 0;
    if (!SAI->getDimensionSize(i)) {
      Array["sizes"].append("*");
      i++;
    }
    for (; i < SAI->getNumberOfDimensions(); i++) {
      SAI->getDimensionSize(i)->print(RawStringOstream);
      Array["sizes"].append(RawStringOstream.str());
      Buffer.clear();
    }
    SAI->getElementType()->print(RawStringOstream);
    Array["type"] = RawStringOstream.str();
    Buffer.clear();
    Arrays.append(Array);
  }
  return Arrays;
}

static Json::Value getJSON(Scop &S) {
  Json::Value root;
  unsigned LineBegin, LineEnd;
  std::string FileName;

  getDebugLocation(&S.getRegion(), LineBegin, LineEnd, FileName);
  std::string Location;
  if (LineBegin != (unsigned)-1)
    Location = FileName + ":" + std::to_string(LineBegin) + "-" +
               std::to_string(LineEnd);

  root["name"] = S.getNameStr();
  root["context"] = S.getContextStr();
  if (LineBegin != (unsigned)-1)
    root["location"] = Location;

  root["arrays"] = exportArrays(S);

  root["statements"];

  for (ScopStmt &Stmt : S) {
    Json::Value statement;

    statement["name"] = Stmt.getBaseName();
    statement["domain"] = Stmt.getDomainStr();
    statement["schedule"] = Stmt.getScheduleStr();
    statement["accesses"];

    for (MemoryAccess *MA : Stmt) {
      Json::Value access;

      access["kind"] = MA->isRead() ? "read" : "write";
      access["relation"] = MA->getAccessRelationStr();

      statement["accesses"].append(access);
    }

    root["statements"].append(statement);
  }

  return root;
}

static void exportScop(Scop &S) {
  std::string FileName = ImportDir + "/" + getFileName(S);

  Json::Value jscop = getJSON(S);
  Json::StyledWriter writer;
  std::string fileContent = writer.write(jscop);

  // Write to file.
  std::error_code EC;
  ToolOutputFile F(FileName, EC, llvm::sys::fs::F_Text);

  std::string FunctionName = S.getFunction().getName();
  errs() << "Writing JScop '" << S.getNameStr() << "' in function '"
         << FunctionName << "' to '" << FileName << "'.\n";

  if (!EC) {
    F.os() << fileContent;
    F.os().close();
    if (!F.os().has_error()) {
      errs() << "\n";
      F.keep();
      return;
    }
  }

  errs() << "  error opening file for writing!\n";
  F.os().clear_error();
}

typedef Dependences::StatementToIslMapTy StatementToIslMapTy;

/// Import a new context from JScop.
///
/// @param S The scop to update.
/// @param JScop The JScop file describing the new schedule.
///
/// @returns True if the import succeeded, otherwise False.
static bool importContext(Scop &S, Json::Value &JScop) {
  isl::set OldContext = S.getContext();

  // Check if key 'context' is present.
  if (!JScop.isMember("context")) {
    errs() << "JScop file has no key named 'context'.\n";
    return false;
  }

  isl::set NewContext =
      isl::set{S.getIslCtx().get(), JScop["context"].asString()};

  // Check whether the context was parsed successfully.
  if (!NewContext) {
    errs() << "The context was not parsed successfully by ISL.\n";
    return false;
  }

  // Check if the isl_set is a parameter set.
  if (!NewContext.is_params()) {
    errs() << "The isl_set is not a parameter set.\n";
    return false;
  }

  unsigned OldContextDim = OldContext.dim(isl::dim::param);
  unsigned NewContextDim = NewContext.dim(isl::dim::param);

  // Check if the imported context has the right number of parameters.
  if (OldContextDim != NewContextDim) {
    errs() << "Imported context has the wrong number of parameters : "
           << "Found " << NewContextDim << " Expected " << OldContextDim
           << "\n";
    return false;
  }

  for (unsigned i = 0; i < OldContextDim; i++) {
    isl::id Id = OldContext.get_dim_id(isl::dim::param, i);
    NewContext = NewContext.set_dim_id(isl::dim::param, i, Id);
  }

  S.setContext(NewContext);
  return true;
}

/// Import a new schedule from JScop.
///
/// ... and verify that the new schedule does preserve existing data
/// dependences.
///
/// @param S The scop to update.
/// @param JScop The JScop file describing the new schedule.
/// @param D The data dependences of the @p S.
///
/// @returns True if the import succeeded, otherwise False.
static bool importSchedule(Scop &S, Json::Value &JScop, const Dependences &D) {
  StatementToIslMapTy NewSchedule;

  // Check if key 'statements' is present.
  if (!JScop.isMember("statements")) {
    errs() << "JScop file has no key name 'statements'.\n";
    return false;
  }

  Json::Value statements = JScop["statements"];

  // Check whether the number of indices equals the number of statements
  if (statements.size() != S.getSize()) {
    errs() << "The number of indices and the number of statements differ.\n";
    return false;
  }

  int Index = 0;
  for (ScopStmt &Stmt : S) {
    // Check if key 'schedule' is present.
    if (!statements[Index].isMember("schedule")) {
      errs() << "Statement " << Index << " has no 'schedule' key.\n";
      for (auto Element : NewSchedule) {
        isl_map_free(Element.second);
      }
      return false;
    }
    Json::Value Schedule = statements[Index]["schedule"];
    assert(!Schedule.asString().empty() &&
           "Schedules that contain extension nodes require special handling.");
    isl_map *Map =
        isl_map_read_from_str(S.getIslCtx().get(), Schedule.asCString());

    // Check whether the schedule was parsed successfully
    if (!Map) {
      errs() << "The schedule was not parsed successfully (index = " << Index
             << ").\n";
      for (auto Element : NewSchedule) {
        isl_map_free(Element.second);
      }
      return false;
    }

    isl_space *Space = Stmt.getDomainSpace().release();

    // Copy the old tuple id. This is necessary to retain the user pointer,
    // that stores the reference to the ScopStmt this schedule belongs to.
    Map = isl_map_set_tuple_id(Map, isl_dim_in,
                               isl_space_get_tuple_id(Space, isl_dim_set));
    for (unsigned i = 0; i < isl_space_dim(Space, isl_dim_param); i++) {
      isl_id *Id = isl_space_get_dim_id(Space, isl_dim_param, i);
      Map = isl_map_set_dim_id(Map, isl_dim_param, i, Id);
    }
    isl_space_free(Space);
    NewSchedule[&Stmt] = Map;
    Index++;
  }

  // Check whether the new schedule is valid or not.
  if (!D.isValidSchedule(S, &NewSchedule)) {
    errs() << "JScop file contains a schedule that changes the "
           << "dependences. Use -disable-polly-legality to continue anyways\n";
    for (auto Element : NewSchedule)
      isl_map_free(Element.second);
    return false;
  }

  auto ScheduleMap = isl::union_map::empty(S.getParamSpace());
  for (ScopStmt &Stmt : S) {
    if (NewSchedule.find(&Stmt) != NewSchedule.end())
      ScheduleMap = ScheduleMap.add_map(isl::manage(NewSchedule[&Stmt]));
    else
      ScheduleMap = ScheduleMap.add_map(Stmt.getSchedule());
  }

  S.setSchedule(ScheduleMap);

  return true;
}

/// Import new memory accesses from JScop.
///
/// @param S The scop to update.
/// @param JScop The JScop file describing the new schedule.
/// @param DL The data layout to assume.
/// @param NewAccessStrings optionally record the imported access strings
///
/// @returns True if the import succeeded, otherwise False.
static bool
importAccesses(Scop &S, Json::Value &JScop, const DataLayout &DL,
               std::vector<std::string> *NewAccessStrings = nullptr) {
  int StatementIdx = 0;

  // Check if key 'statements' is present.
  if (!JScop.isMember("statements")) {
    errs() << "JScop file has no key name 'statements'.\n";
    return false;
  }
  Json::Value statements = JScop["statements"];

  // Check whether the number of indices equals the number of statements
  if (statements.size() != S.getSize()) {
    errs() << "The number of indices and the number of statements differ.\n";
    return false;
  }

  for (ScopStmt &Stmt : S) {
    int MemoryAccessIdx = 0;

    // Check if key 'accesses' is present.
    if (!statements[StatementIdx].isMember("accesses")) {
      errs()
          << "Statement from JScop file has no key name 'accesses' for index "
          << StatementIdx << ".\n";
      return false;
    }

    // Check whether the number of indices equals the number of memory
    // accesses
    if (Stmt.size() != statements[StatementIdx]["accesses"].size()) {
      errs() << "The number of memory accesses in the JSop file and the number "
                "of memory accesses differ for index "
             << StatementIdx << ".\n";
      return false;
    }

    for (MemoryAccess *MA : Stmt) {
      // Check if key 'relation' is present.
      Json::Value JsonMemoryAccess =
          statements[StatementIdx]["accesses"][MemoryAccessIdx];
      if (!JsonMemoryAccess.isMember("relation")) {
        errs() << "Memory access number " << MemoryAccessIdx
               << " has no key name 'relation' for statement number "
               << StatementIdx << ".\n";
        return false;
      }
      Json::Value Accesses = JsonMemoryAccess["relation"];
      isl_map *NewAccessMap =
          isl_map_read_from_str(S.getIslCtx().get(), Accesses.asCString());

      // Check whether the access was parsed successfully
      if (!NewAccessMap) {
        errs() << "The access was not parsed successfully by ISL.\n";
        return false;
      }
      isl_map *CurrentAccessMap = MA->getAccessRelation().release();

      // Check if the number of parameter change
      if (isl_map_dim(NewAccessMap, isl_dim_param) !=
          isl_map_dim(CurrentAccessMap, isl_dim_param)) {
        errs() << "JScop file changes the number of parameter dimensions.\n";
        isl_map_free(CurrentAccessMap);
        isl_map_free(NewAccessMap);
        return false;
      }

      isl_id *NewOutId;

      // If the NewAccessMap has zero dimensions, it is the scalar access; it
      // must be the same as before.
      // If it has at least one dimension, it's an array access; search for
      // its ScopArrayInfo.
      if (isl_map_dim(NewAccessMap, isl_dim_out) >= 1) {
        NewOutId = isl_map_get_tuple_id(NewAccessMap, isl_dim_out);
        auto *SAI = S.getArrayInfoByName(isl_id_get_name(NewOutId));
        isl_id *OutId = isl_map_get_tuple_id(CurrentAccessMap, isl_dim_out);
        auto *OutSAI = ScopArrayInfo::getFromId(isl::manage(OutId));
        if (!SAI || SAI->getElementType() != OutSAI->getElementType()) {
          errs() << "JScop file contains access function with undeclared "
                    "ScopArrayInfo\n";
          isl_map_free(CurrentAccessMap);
          isl_map_free(NewAccessMap);
          isl_id_free(NewOutId);
          return false;
        }
        isl_id_free(NewOutId);
        NewOutId = SAI->getBasePtrId().release();
      } else {
        NewOutId = isl_map_get_tuple_id(CurrentAccessMap, isl_dim_out);
      }

      NewAccessMap = isl_map_set_tuple_id(NewAccessMap, isl_dim_out, NewOutId);

      if (MA->isArrayKind()) {
        // We keep the old alignment, thus we cannot allow accesses to memory
        // locations that were not accessed before if the alignment of the
        // access is not the default alignment.
        bool SpecialAlignment = true;
        if (LoadInst *LoadI = dyn_cast<LoadInst>(MA->getAccessInstruction())) {
          SpecialAlignment =
              LoadI->getAlignment() &&
              DL.getABITypeAlignment(LoadI->getType()) != LoadI->getAlignment();
        } else if (StoreInst *StoreI =
                       dyn_cast<StoreInst>(MA->getAccessInstruction())) {
          SpecialAlignment =
              StoreI->getAlignment() &&
              DL.getABITypeAlignment(StoreI->getValueOperand()->getType()) !=
                  StoreI->getAlignment();
        }

        if (SpecialAlignment) {
          isl_set *NewAccessSet = isl_map_range(isl_map_copy(NewAccessMap));
          isl_set *CurrentAccessSet =
              isl_map_range(isl_map_copy(CurrentAccessMap));
          bool IsSubset = isl_set_is_subset(NewAccessSet, CurrentAccessSet);
          isl_set_free(NewAccessSet);
          isl_set_free(CurrentAccessSet);

          // Check if the JScop file changes the accessed memory.
          if (!IsSubset) {
            errs() << "JScop file changes the accessed memory\n";
            isl_map_free(CurrentAccessMap);
            isl_map_free(NewAccessMap);
            return false;
          }
        }
      }

      // We need to copy the isl_ids for the parameter dimensions to the new
      // map. Without doing this the current map would have different
      // ids then the new one, even though both are named identically.
      for (unsigned i = 0; i < isl_map_dim(CurrentAccessMap, isl_dim_param);
           i++) {
        isl_id *Id = isl_map_get_dim_id(CurrentAccessMap, isl_dim_param, i);
        NewAccessMap = isl_map_set_dim_id(NewAccessMap, isl_dim_param, i, Id);
      }

      // Copy the old tuple id. This is necessary to retain the user pointer,
      // that stores the reference to the ScopStmt this access belongs to.
      isl_id *Id = isl_map_get_tuple_id(CurrentAccessMap, isl_dim_in);
      NewAccessMap = isl_map_set_tuple_id(NewAccessMap, isl_dim_in, Id);

      auto NewAccessDomain = isl_map_domain(isl_map_copy(NewAccessMap));
      auto CurrentAccessDomain = isl_map_domain(isl_map_copy(CurrentAccessMap));

      if (!isl_set_has_equal_space(NewAccessDomain, CurrentAccessDomain)) {
        errs() << "JScop file contains access function with incompatible "
               << "dimensions\n";
        isl_map_free(CurrentAccessMap);
        isl_map_free(NewAccessMap);
        isl_set_free(NewAccessDomain);
        isl_set_free(CurrentAccessDomain);
        return false;
      }

      NewAccessDomain =
          isl_set_intersect_params(NewAccessDomain, S.getContext().release());
      CurrentAccessDomain = isl_set_intersect_params(CurrentAccessDomain,
                                                     S.getContext().release());
      CurrentAccessDomain =
          isl_set_intersect(CurrentAccessDomain, Stmt.getDomain().release());

      if (MA->isRead() &&
          isl_set_is_subset(CurrentAccessDomain, NewAccessDomain) ==
              isl_bool_false) {
        errs() << "Mapping not defined for all iteration domain elements\n";
        isl_set_free(CurrentAccessDomain);
        isl_set_free(NewAccessDomain);
        isl_map_free(CurrentAccessMap);
        isl_map_free(NewAccessMap);
        return false;
      }

      isl_set_free(CurrentAccessDomain);
      isl_set_free(NewAccessDomain);

      if (!isl_map_is_equal(NewAccessMap, CurrentAccessMap)) {
        // Statistics.
        ++NewAccessMapFound;
        if (NewAccessStrings)
          NewAccessStrings->push_back(Accesses.asCString());
        MA->setNewAccessRelation(isl::manage(NewAccessMap));
      } else {
        isl_map_free(NewAccessMap);
      }
      isl_map_free(CurrentAccessMap);
      MemoryAccessIdx++;
    }
    StatementIdx++;
  }

  return true;
}

/// Check whether @p SAI and @p Array represent the same array.
static bool areArraysEqual(ScopArrayInfo *SAI, Json::Value Array) {
  std::string Buffer;
  llvm::raw_string_ostream RawStringOstream(Buffer);

  // Check if key 'type' is present.
  if (!Array.isMember("type")) {
    errs() << "Array has no key 'type'.\n";
    return false;
  }

  // Check if key 'sizes' is present.
  if (!Array.isMember("sizes")) {
    errs() << "Array has no key 'sizes'.\n";
    return false;
  }

  // Check if key 'name' is present.
  if (!Array.isMember("name")) {
    errs() << "Array has no key 'name'.\n";
    return false;
  }

  if (SAI->getName() != Array["name"].asCString())
    return false;

  if (SAI->getNumberOfDimensions() != Array["sizes"].size())
    return false;

  for (unsigned i = 1; i < Array["sizes"].size(); i++) {
    SAI->getDimensionSize(i)->print(RawStringOstream);
    if (RawStringOstream.str() != Array["sizes"][i].asCString())
      return false;
    Buffer.clear();
  }

  // Check if key 'type' differs from the current one or is not valid.
  SAI->getElementType()->print(RawStringOstream);
  if (RawStringOstream.str() != Array["type"].asCString()) {
    errs() << "Array has not a valid type.\n";
    return false;
  }

  return true;
}

/// Get the accepted primitive type from its textual representation
///        @p TypeTextRepresentation.
///
/// @param TypeTextRepresentation The textual representation of the type.
/// @return The pointer to the primitive type, if this type is accepted
///         or nullptr otherwise.
static Type *parseTextType(const std::string &TypeTextRepresentation,
                           LLVMContext &LLVMContext) {
  std::map<std::string, Type *> MapStrToType = {
      {"void", Type::getVoidTy(LLVMContext)},
      {"half", Type::getHalfTy(LLVMContext)},
      {"float", Type::getFloatTy(LLVMContext)},
      {"double", Type::getDoubleTy(LLVMContext)},
      {"x86_fp80", Type::getX86_FP80Ty(LLVMContext)},
      {"fp128", Type::getFP128Ty(LLVMContext)},
      {"ppc_fp128", Type::getPPC_FP128Ty(LLVMContext)},
      {"i1", Type::getInt1Ty(LLVMContext)},
      {"i8", Type::getInt8Ty(LLVMContext)},
      {"i16", Type::getInt16Ty(LLVMContext)},
      {"i32", Type::getInt32Ty(LLVMContext)},
      {"i64", Type::getInt64Ty(LLVMContext)},
      {"i128", Type::getInt128Ty(LLVMContext)}};

  auto It = MapStrToType.find(TypeTextRepresentation);
  if (It != MapStrToType.end())
    return It->second;

  errs() << "Textual representation can not be parsed: "
         << TypeTextRepresentation << "\n";
  return nullptr;
}

/// Import new arrays from JScop.
///
/// @param S The scop to update.
/// @param JScop The JScop file describing new arrays.
///
/// @returns True if the import succeeded, otherwise False.
static bool importArrays(Scop &S, Json::Value &JScop) {
  Json::Value Arrays = JScop["arrays"];
  if (Arrays.size() == 0)
    return true;

  unsigned ArrayIdx = 0;
  for (auto &SAI : S.arrays()) {
    if (!SAI->isArrayKind())
      continue;
    if (ArrayIdx + 1 > Arrays.size()) {
      errs() << "Not enough array entries in JScop file.\n";
      return false;
    }
    if (!areArraysEqual(SAI, Arrays[ArrayIdx])) {
      errs() << "No match for array '" << SAI->getName() << "' in JScop.\n";
      return false;
    }
    ArrayIdx++;
  }

  for (; ArrayIdx < Arrays.size(); ArrayIdx++) {
    auto &Array = Arrays[ArrayIdx];
    auto *ElementType =
        parseTextType(Array["type"].asCString(), S.getSE()->getContext());
    if (!ElementType) {
      errs() << "Error while parsing element type for new array.\n";
      return false;
    }
    std::vector<unsigned> DimSizes;
    for (unsigned i = 0; i < Array["sizes"].size(); i++) {
      auto Size = std::stoi(Array["sizes"][i].asCString());

      // Check if the size if positive.
      if (Size <= 0) {
        errs() << "The size at index " << i << " is =< 0.\n";
        return false;
      }

      DimSizes.push_back(Size);
    }

    auto NewSAI =
        S.createScopArrayInfo(ElementType, Array["name"].asCString(), DimSizes);

    if (Array.isMember("allocation")) {
      NewSAI->setIsOnHeap(Array["allocation"].asString() == "heap");
    }
  }

  return true;
}

/// Import a Scop from a JSCOP file
/// @param S The scop to be modified
/// @param D Dependence Info
/// @param DL The DataLayout of the function
/// @param NewAccessStrings Optionally record the imported access strings
///
/// @returns true on success, false otherwise. Beware that if this returns
/// false, the Scop may still have been modified. In this case the Scop contains
/// invalid information.
static bool importScop(Scop &S, const Dependences &D, const DataLayout &DL,
                       std::vector<std::string> *NewAccessStrings = nullptr) {
  std::string FileName = ImportDir + "/" + getFileName(S, ImportPostfix);

  std::string FunctionName = S.getFunction().getName();
  errs() << "Reading JScop '" << S.getNameStr() << "' in function '"
         << FunctionName << "' from '" << FileName << "'.\n";
  ErrorOr<std::unique_ptr<MemoryBuffer>> result =
      MemoryBuffer::getFile(FileName);
  std::error_code ec = result.getError();

  if (ec) {
    errs() << "File could not be read: " << ec.message() << "\n";
    return false;
  }

  Json::Reader reader;
  Json::Value jscop;

  bool parsingSuccessful = reader.parse(result.get()->getBufferStart(), jscop);

  if (!parsingSuccessful) {
    errs() << "JSCoP file could not be parsed\n";
    return false;
  }

  bool Success = importContext(S, jscop);

  if (!Success)
    return false;

  Success = importSchedule(S, jscop, D);

  if (!Success)
    return false;

  Success = importArrays(S, jscop);

  if (!Success)
    return false;

  Success = importAccesses(S, jscop, DL, NewAccessStrings);

  if (!Success)
    return false;
  return true;
}

char JSONExporter::ID = 0;
void JSONExporter::printScop(raw_ostream &OS, Scop &S) const { OS << S; }

bool JSONExporter::runOnScop(Scop &S) {
  exportScop(S);
  return false;
}

void JSONExporter::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<ScopInfoRegionPass>();
}

Pass *polly::createJSONExporterPass() { return new JSONExporter(); }

PreservedAnalyses JSONExportPass::run(Scop &S, ScopAnalysisManager &SAM,
                                      ScopStandardAnalysisResults &SAR,
                                      SPMUpdater &) {
  exportScop(S);
  return PreservedAnalyses::all();
}

char JSONImporter::ID = 0;

void JSONImporter::printScop(raw_ostream &OS, Scop &S) const {
  OS << S;
  for (std::vector<std::string>::const_iterator I = NewAccessStrings.begin(),
                                                E = NewAccessStrings.end();
       I != E; I++)
    OS << "New access function '" << *I << "' detected in JSCOP file\n";
}

bool JSONImporter::runOnScop(Scop &S) {
  const Dependences &D =
      getAnalysis<DependenceInfo>().getDependences(Dependences::AL_Statement);
  const DataLayout &DL = S.getFunction().getParent()->getDataLayout();

  if (!importScop(S, D, DL, &NewAccessStrings))
    report_fatal_error("Tried to import a malformed jscop file.");

  return false;
}

void JSONImporter::getAnalysisUsage(AnalysisUsage &AU) const {
  ScopPass::getAnalysisUsage(AU);
  AU.addRequired<DependenceInfo>();

  // TODO: JSONImporter should throw away DependenceInfo.
  AU.addPreserved<DependenceInfo>();
}

Pass *polly::createJSONImporterPass() { return new JSONImporter(); }

PreservedAnalyses JSONImportPass::run(Scop &S, ScopAnalysisManager &SAM,
                                      ScopStandardAnalysisResults &SAR,
                                      SPMUpdater &) {
  const Dependences &D =
      SAM.getResult<DependenceAnalysis>(S, SAR).getDependences(
          Dependences::AL_Statement);
  const DataLayout &DL = S.getFunction().getParent()->getDataLayout();

  if (!importScop(S, D, DL))
    report_fatal_error("Tried to import a malformed jscop file.");

  // This invalidates all analyses on Scop.
  PreservedAnalyses PA;
  PA.preserveSet<AllAnalysesOn<Module>>();
  PA.preserveSet<AllAnalysesOn<Function>>();
  PA.preserveSet<AllAnalysesOn<Loop>>();
  return PA;
}

INITIALIZE_PASS_BEGIN(JSONExporter, "polly-export-jscop",
                      "Polly - Export Scops as JSON"
                      " (Writes a .jscop file for each Scop)",
                      false, false);
INITIALIZE_PASS_DEPENDENCY(DependenceInfo)
INITIALIZE_PASS_END(JSONExporter, "polly-export-jscop",
                    "Polly - Export Scops as JSON"
                    " (Writes a .jscop file for each Scop)",
                    false, false)

INITIALIZE_PASS_BEGIN(JSONImporter, "polly-import-jscop",
                      "Polly - Import Scops from JSON"
                      " (Reads a .jscop file for each Scop)",
                      false, false);
INITIALIZE_PASS_DEPENDENCY(DependenceInfo)
INITIALIZE_PASS_END(JSONImporter, "polly-import-jscop",
                    "Polly - Import Scops from JSON"
                    " (Reads a .jscop file for each Scop)",
                    false, false)
