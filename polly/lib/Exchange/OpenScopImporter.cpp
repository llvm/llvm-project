//===-- OpenScopImporter.cpp  - Import Scops with openscop library --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Import modified .scop files into Polly. This allows to change the schedule of
// statements.
//
//===----------------------------------------------------------------------===//

#include "polly/LinkAllPasses.h"
#include "polly/Dependences.h"
#include "polly/Options.h"
#include "polly/ScopInfo.h"
#include "polly/ScopPass.h"
#include "llvm/Assembly/Writer.h"

#ifdef OPENSCOP_FOUND

#define OPENSCOP_INT_T_IS_MP
#include "openscop/openscop.h"

#include "isl/map.h"
#include "isl/set.h"
#include "isl/constraint.h"

using namespace llvm;
using namespace polly;

namespace {
static cl::opt<std::string>
ImportDir("polly-import-dir",
          cl::desc("The directory to import the .scop files from."), cl::Hidden,
          cl::value_desc("Directory path"), cl::ValueRequired, cl::init("."));
static cl::opt<std::string>
ImportPostfix("polly-import-postfix",
              cl::desc("Postfix to append to the import .scop files."),
              cl::Hidden, cl::value_desc("File postfix"), cl::ValueRequired,
              cl::init(""));

struct ScopImporter : public ScopPass {
  static char ID;
  Scop *S;
  Dependences *D;
  explicit ScopImporter() : ScopPass(ID) {}
  bool updateScattering(Scop *S, openscop_scop_p OScop);

  std::string getFileName(Scop *S) const;
  virtual bool runOnScop(Scop &S);
  virtual void printScop(raw_ostream &OS) const;
  void getAnalysisUsage(AnalysisUsage &AU) const;
};
}

char ScopImporter::ID = 0;

/// @brief Create an isl constraint from a row of OpenScop integers.
///
/// @param row An array of isl/OpenScop integers.
/// @param Space An isl space object, describing how to spilt the dimensions.
///
/// @return An isl constraint representing this integer array.
isl_constraint *constraintFromMatrixRow(isl_int *row, isl_space *Space) {
  isl_constraint *c;

  unsigned NbOut = isl_space_size(Space, isl_dim_out);
  unsigned NbIn = isl_space_size(Space, isl_dim_in);
  unsigned NbParam = isl_space_size(Space, isl_dim_param);

  if (isl_int_is_zero(row[0]))
    c = isl_equality_alloc(isl_space_copy(Space));
  else
    c = isl_inequality_alloc(isl_space_copy(Space));

  unsigned current_column = 1;

  for (unsigned j = 0; j < NbOut; ++j)
    isl_constraint_set_coefficient(c, isl_space_out, j, row[current_column++]);

  for (unsigned j = 0; j < NbIn; ++j)
    isl_constraint_set_coefficient(c, isl_space_in, j, row[current_column++]);

  for (unsigned j = 0; j < NbParam; ++j)
    isl_constraint_set_coefficient(c, isl_space_param, j,
                                   row[current_column++]);

  isl_constraint_set_constant(c, row[current_column]);

  return c;
}

/// @brief Create an isl map from a OpenScop matrix.
///
/// @param m The OpenScop matrix to translate.
/// @param Space The dimensions that are contained in the OpenScop matrix.
///
/// @return An isl map representing m.
isl_map *mapFromMatrix(openscop_matrix_p m, isl_space *Space) {
  isl_basic_map *bmap = isl_basic_map_universe(isl_space_copy(Space));

  for (unsigned i = 0; i < m->NbRows; ++i) {
    isl_constraint *c;

    c = constraintFromMatrixRow(m->p[i], Space);
    bmap = isl_basic_map_add_constraint(bmap, c);
  }

  return isl_map_from_basic_map(bmap);
}

/// @brief Create a new scattering for PollyStmt.
///
/// @param m The matrix describing the new scattering.
/// @param PollyStmt The statement to create the scattering for.
///
/// @return An isl_map describing the scattering.
isl_map *scatteringForStmt(openscop_matrix_p m, ScopStmt *PollyStmt) {

  unsigned NbParam = PollyStmt->getNumParams();
  unsigned NbIterators = PollyStmt->getNumIterators();
  unsigned NbScattering = m->NbColumns - 2 - NbParam - NbIterators;

  isl_ctx *ctx = PollyStmt->getParent()->getCtx();
  isl_space *Space = isl_dim_alloc(ctx, NbParam, NbIterators, NbScattering);
  Space = isl_space_set_tuple_name(Space, isl_dim_out, "scattering");
  Space = isl_space_set_tuple_name(Space, isl_dim_in, PollyStmt->getBaseName());
  isl_map *map = mapFromMatrix(m, Space);
  isl_space_free(Space);

  return map;
}

typedef Dependences::StatementToIslMapTy StatementToIslMapTy;

/// @brief Read the new scattering from the OpenScop description.
///
/// @S      The Scop to update
/// @OScop  The OpenScop data structure describing the new scattering.
/// @return A map that contains for each Statement the new scattering.
StatementToIslMapTy *readScattering(Scop *S, openscop_scop_p OScop) {
  StatementToIslMapTy &NewScattering = *(new StatementToIslMapTy());
  openscop_statement_p stmt = OScop->statement;

  for (Scop::iterator SI = S->begin(), SE = S->end(); SI != SE; ++SI) {
    if (!stmt) {
      errs() << "Not enough statements available in OpenScop file\n";
      delete &NewScattering;
      return NULL;
    }

    NewScattering[*SI] = scatteringForStmt(stmt->schedule, *SI);
    stmt = stmt->next;
  }

  if (stmt) {
    errs() << "Too many statements in OpenScop file\n";
    delete &NewScattering;
    return NULL;
  }

  return &NewScattering;
}

/// @brief Update the scattering in a Scop using the OpenScop description of
/// the scattering.
///
/// @S The Scop to update
/// @OScop The OpenScop data structure describing the new scattering.
/// @return Returns false, if the update failed.
bool ScopImporter::updateScattering(Scop *S, openscop_scop_p OScop) {
  StatementToIslMapTy *NewScattering = readScattering(S, OScop);

  if (!NewScattering)
    return false;

  if (!D->isValidScattering(NewScattering)) {
    errs() << "OpenScop file contains a scattering that changes the "
           << "dependences. Use -disable-polly-legality to continue anyways\n";
    return false;
  }

  for (Scop::iterator SI = S->begin(), SE = S->end(); SI != SE; ++SI) {
    ScopStmt *Stmt = *SI;

    if (NewScattering->find(Stmt) != NewScattering->end())
      Stmt->setScattering((*NewScattering)[Stmt]);
  }

  return true;
}
std::string ScopImporter::getFileName(Scop *S) const {
  std::string FunctionName = S->getRegion().getEntry()->getParent()->getName();
  std::string FileName = FunctionName + "___" + S->getNameStr() + ".scop";
  return FileName;
}

void ScopImporter::printScop(raw_ostream &OS) const { S->print(OS); }

bool ScopImporter::runOnScop(Scop &scop) {
  S = &scop;
  Region &R = scop.getRegion();
  D = &getAnalysis<Dependences>();

  std::string FileName = ImportDir + "/" + getFileName(S) + ImportPostfix;
  FILE *F = fopen(FileName.c_str(), "r");

  if (!F) {
    errs() << "Cannot open file: " << FileName << "\n";
    errs() << "Skipping import.\n";
    return false;
  }

  openscop_scop_p openscop = openscop_scop_read(F);
  fclose(F);

  std::string FunctionName = R.getEntry()->getParent()->getName();
  errs() << "Reading Scop '" << R.getNameStr() << "' in function '"
         << FunctionName << "' from '" << FileName << "'.\n";

  bool UpdateSuccessfull = updateScattering(S, openscop);

  if (!UpdateSuccessfull) {
    errs() << "Update failed"
           << "\n";
  }

  return false;
}

void ScopImporter::getAnalysisUsage(AnalysisUsage &AU) const {
  ScopPass::getAnalysisUsage(AU);
  AU.addRequired<Dependences>();
}

static RegisterPass<ScopImporter> A("polly-import",
                                    "Polly - Import Scops with OpenScop library"
                                    " (Reads a .scop file for each Scop)");

Pass *polly::createScopImporterPass() { return new ScopImporter(); }

#endif
