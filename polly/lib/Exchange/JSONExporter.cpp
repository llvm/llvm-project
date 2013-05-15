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

#include "polly/LinkAllPasses.h"
#include "polly/Dependences.h"
#include "polly/Options.h"
#include "polly/ScopInfo.h"
#include "polly/ScopPass.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/system_error.h"

#define DEBUG_TYPE "polly-import-jscop"

#include "json/reader.h"
#include "json/writer.h"

#include "isl/set.h"
#include "isl/map.h"
#include "isl/constraint.h"
#include "isl/printer.h"

#include <string>

using namespace llvm;
using namespace polly;

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
  Scop *S;
  explicit JSONExporter() : ScopPass(ID) {}

  std::string getFileName(Scop *S) const;
  Json::Value getJSON(Scop &scop) const;
  virtual bool runOnScop(Scop &S);
  void printScop(raw_ostream &OS) const;
  void getAnalysisUsage(AnalysisUsage &AU) const;
};

struct JSONImporter : public ScopPass {
  static char ID;
  Scop *S;
  std::vector<std::string> newAccessStrings;
  explicit JSONImporter() : ScopPass(ID) {}

  std::string getFileName(Scop *S) const;
  virtual bool runOnScop(Scop &S);
  void printScop(raw_ostream &OS) const;
  void getAnalysisUsage(AnalysisUsage &AU) const;
};

}

char JSONExporter::ID = 0;
std::string JSONExporter::getFileName(Scop *S) const {
  std::string FunctionName = S->getRegion().getEntry()->getParent()->getName();
  std::string FileName = FunctionName + "___" + S->getNameStr() + ".jscop";
  return FileName;
}

void JSONExporter::printScop(raw_ostream &OS) const { S->print(OS); }

Json::Value JSONExporter::getJSON(Scop &scop) const {
  Json::Value root;

  root["name"] = S->getRegion().getNameStr();
  root["context"] = S->getContextStr();
  root["statements"];

  for (Scop::iterator SI = S->begin(), SE = S->end(); SI != SE; ++SI) {
    ScopStmt *Stmt = *SI;

    Json::Value statement;

    statement["name"] = Stmt->getBaseName();
    statement["domain"] = Stmt->getDomainStr();
    statement["schedule"] = Stmt->getScatteringStr();
    statement["accesses"];

    for (ScopStmt::memacc_iterator MI = Stmt->memacc_begin(),
                                   ME = Stmt->memacc_end();
         MI != ME; ++MI) {
      Json::Value access;

      access["kind"] = (*MI)->isRead() ? "read" : "write";
      access["relation"] = (*MI)->getAccessRelationStr();

      statement["accesses"].append(access);
    }

    root["statements"].append(statement);
  }

  return root;
}

bool JSONExporter::runOnScop(Scop &scop) {
  S = &scop;
  Region &R = S->getRegion();

  std::string FileName = ImportDir + "/" + getFileName(S);

  Json::Value jscop = getJSON(scop);
  Json::StyledWriter writer;
  std::string fileContent = writer.write(jscop);

  // Write to file.
  std::string ErrInfo;
  tool_output_file F(FileName.c_str(), ErrInfo);

  std::string FunctionName = R.getEntry()->getParent()->getName();
  errs() << "Writing JScop '" << R.getNameStr() << "' in function '"
         << FunctionName << "' to '" << FileName << "'.\n";

  if (ErrInfo.empty()) {
    F.os() << fileContent;
    F.os().close();
    if (!F.os().has_error()) {
      errs() << "\n";
      F.keep();
      return false;
    }
  }

  errs() << "  error opening file for writing!\n";
  F.os().clear_error();

  return false;
}

void JSONExporter::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<ScopInfo>();
}

Pass *polly::createJSONExporterPass() { return new JSONExporter(); }

char JSONImporter::ID = 0;
std::string JSONImporter::getFileName(Scop *S) const {
  std::string FunctionName = S->getRegion().getEntry()->getParent()->getName();
  std::string FileName = FunctionName + "___" + S->getNameStr() + ".jscop";

  if (ImportPostfix != "")
    FileName += "." + ImportPostfix;

  return FileName;
}

void JSONImporter::printScop(raw_ostream &OS) const {
  S->print(OS);
  for (std::vector<std::string>::const_iterator I = newAccessStrings.begin(),
                                                E = newAccessStrings.end();
       I != E; I++)
    OS << "New access function '" << *I << "'detected in JSCOP file\n";
}

typedef Dependences::StatementToIslMapTy StatementToIslMapTy;

bool JSONImporter::runOnScop(Scop &scop) {
  S = &scop;
  Region &R = S->getRegion();
  Dependences *D = &getAnalysis<Dependences>();

  std::string FileName = ImportDir + "/" + getFileName(S);

  std::string FunctionName = R.getEntry()->getParent()->getName();
  errs() << "Reading JScop '" << R.getNameStr() << "' in function '"
         << FunctionName << "' from '" << FileName << "'.\n";
  OwningPtr<MemoryBuffer> result;
  error_code ec = MemoryBuffer::getFile(FileName, result);

  if (ec) {
    errs() << "File could not be read: " << ec.message() << "\n";
    return false;
  }

  Json::Reader reader;
  Json::Value jscop;

  bool parsingSuccessful = reader.parse(result->getBufferStart(), jscop);

  if (!parsingSuccessful) {
    errs() << "JSCoP file could not be parsed\n";
    return false;
  }

  isl_set *OldContext = S->getContext();
  isl_set *NewContext =
      isl_set_read_from_str(S->getIslCtx(), jscop["context"].asCString());

  for (unsigned i = 0; i < isl_set_dim(OldContext, isl_dim_param); i++) {
    isl_id *id = isl_set_get_dim_id(OldContext, isl_dim_param, i);
    NewContext = isl_set_set_dim_id(NewContext, isl_dim_param, i, id);

  }

  isl_set_free(OldContext);
  S->setContext(NewContext);

  StatementToIslMapTy &NewScattering = *(new StatementToIslMapTy());

  int index = 0;

  for (Scop::iterator SI = S->begin(), SE = S->end(); SI != SE; ++SI) {
    Json::Value schedule = jscop["statements"][index]["schedule"];
    isl_map *m = isl_map_read_from_str(S->getIslCtx(), schedule.asCString());
    isl_space *Space = (*SI)->getDomainSpace();

    // Copy the old tuple id. This is necessary to retain the user pointer,
    // that stores the reference to the ScopStmt this scattering belongs to.
    m = isl_map_set_tuple_id(m, isl_dim_in,
                             isl_space_get_tuple_id(Space, isl_dim_set));
    isl_space_free(Space);
    NewScattering[*SI] = m;
    index++;
  }

  if (!D->isValidScattering(&NewScattering)) {
    errs() << "JScop file contains a scattering that changes the "
           << "dependences. Use -disable-polly-legality to continue anyways\n";
    return false;
  }

  for (Scop::iterator SI = S->begin(), SE = S->end(); SI != SE; ++SI) {
    ScopStmt *Stmt = *SI;

    if (NewScattering.find(Stmt) != NewScattering.end())
      Stmt->setScattering(NewScattering[Stmt]);
  }

  int statementIdx = 0;
  for (Scop::iterator SI = S->begin(), SE = S->end(); SI != SE; ++SI) {
    ScopStmt *Stmt = *SI;

    int memoryAccessIdx = 0;
    for (ScopStmt::memacc_iterator MI = Stmt->memacc_begin(),
                                   ME = Stmt->memacc_end();
         MI != ME; ++MI) {
      Json::Value accesses = jscop["statements"][statementIdx]["accesses"][
          memoryAccessIdx]["relation"];
      isl_map *newAccessMap =
          isl_map_read_from_str(S->getIslCtx(), accesses.asCString());
      isl_map *currentAccessMap = (*MI)->getAccessRelation();

      if (isl_map_dim(newAccessMap, isl_dim_param) !=
          isl_map_dim(currentAccessMap, isl_dim_param)) {
        errs() << "JScop file changes the number of parameter dimensions\n";
        isl_map_free(currentAccessMap);
        isl_map_free(newAccessMap);
        return false;

      }

      // We need to copy the isl_ids for the parameter dimensions to the new
      // map. Without doing this the current map would have different
      // ids then the new one, even though both are named identically.
      for (unsigned i = 0; i < isl_map_dim(currentAccessMap, isl_dim_param);
           i++) {
        isl_id *id = isl_map_get_dim_id(currentAccessMap, isl_dim_param, i);
        newAccessMap = isl_map_set_dim_id(newAccessMap, isl_dim_param, i, id);
      }

      // Copy the old tuple id. This is necessary to retain the user pointer,
      // that stores the reference to the ScopStmt this access belongs to.
      isl_id *Id = isl_map_get_tuple_id(currentAccessMap, isl_dim_in);
      newAccessMap = isl_map_set_tuple_id(newAccessMap, isl_dim_in, Id);

      if (!isl_map_has_equal_space(currentAccessMap, newAccessMap)) {
        errs() << "JScop file contains access function with incompatible "
               << "dimensions\n";
        isl_map_free(currentAccessMap);
        isl_map_free(newAccessMap);
        return false;
      }
      if (isl_map_dim(newAccessMap, isl_dim_out) != 1) {
        errs() << "New access map in JScop file should be single dimensional\n";
        isl_map_free(currentAccessMap);
        isl_map_free(newAccessMap);
        return false;
      }
      if (!isl_map_is_equal(newAccessMap, currentAccessMap)) {
        // Statistics.
        ++NewAccessMapFound;
        newAccessStrings.push_back(accesses.asCString());
        (*MI)->setNewAccessRelation(newAccessMap);
      } else {
        isl_map_free(newAccessMap);
      }
      isl_map_free(currentAccessMap);
      memoryAccessIdx++;
    }
    statementIdx++;
  }

  return false;
}

void JSONImporter::getAnalysisUsage(AnalysisUsage &AU) const {
  ScopPass::getAnalysisUsage(AU);
  AU.addRequired<Dependences>();
}
Pass *polly::createJSONImporterPass() { return new JSONImporter(); }

INITIALIZE_PASS_BEGIN(JSONExporter, "polly-export-jscop",
                      "Polly - Export Scops as JSON"
                      " (Writes a .jscop file for each Scop)",
                      false, false);
INITIALIZE_PASS_DEPENDENCY(Dependences)
INITIALIZE_PASS_END(JSONExporter, "polly-export-jscop",
                    "Polly - Export Scops as JSON"
                    " (Writes a .jscop file for each Scop)",
                    false, false)

INITIALIZE_PASS_BEGIN(JSONImporter, "polly-import-jscop",
                      "Polly - Import Scops from JSON"
                      " (Reads a .jscop file for each Scop)",
                      false, false);
INITIALIZE_PASS_DEPENDENCY(Dependences)
INITIALIZE_PASS_END(JSONImporter, "polly-import-jscop",
                    "Polly - Import Scops from JSON"
                    " (Reads a .jscop file for each Scop)",
                    false, false)
