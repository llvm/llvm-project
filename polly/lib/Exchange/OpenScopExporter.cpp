//===-- OpenScopExporter.cpp  - Export Scops with openscop library --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Export the Scops build by ScopInfo pass to text file.
//
//===----------------------------------------------------------------------===//

#include "polly/LinkAllPasses.h"

#ifdef OPENSCOP_FOUND

#include "polly/Options.h"
#include "polly/ScopInfo.h"
#include "polly/ScopPass.h"

#include "llvm/Assembly/Writer.h"

#define OPENSCOP_INT_T_IS_MP
#include "openscop/openscop.h"

#include "stdio.h"
#include "isl/map.h"
#include "isl/set.h"
#include "isl/constraint.h"

using namespace llvm;
using namespace polly;

namespace {
static cl::opt<std::string>
ExportDir("polly-export-dir",
          cl::desc("The directory to export the .scop files to."), cl::Hidden,
          cl::value_desc("Directory path"), cl::ValueRequired, cl::init("."),
          cl::cat(PollyCategory));

struct ScopExporter : public ScopPass {
  static char ID;
  Scop *S;
  explicit ScopExporter() : ScopPass(ID) {}

  std::string getFileName(Scop *S) const;

  virtual bool runOnScop(Scop &S);
  void printScop(raw_ostream &OS) const;
  void getAnalysisUsage(AnalysisUsage &AU) const;
};

}

char ScopExporter::ID = 0;

class OpenScop {
  Scop *PollyScop;
  openscop_scop_p openscop;

  std::map<const Value *, int> ArrayMap;

  void initializeArrays();
  void initializeParameters();
  void initializeScattering();
  void initializeStatements();
  openscop_statement_p initializeStatement(ScopStmt *stmt);
  void freeStatement(openscop_statement_p stmt);
  static int accessToMatrix_constraint(isl_constraint *c, void *user);
  static int accessToMatrix_basic_map(isl_basic_map *bmap, void *user);
  openscop_matrix_p createAccessMatrix(ScopStmt *S, bool isRead);
  static int domainToMatrix_constraint(isl_constraint *c, void *user);
  static int domainToMatrix_basic_set(isl_basic_set *bset, void *user);
  openscop_matrix_p domainToMatrix(isl_set *PS);
  static int scatteringToMatrix_constraint(isl_constraint *c, void *user);
  static int scatteringToMatrix_basic_map(isl_basic_map *bmap, void *user);
  openscop_matrix_p scatteringToMatrix(isl_map *pmap);

public:
  OpenScop(Scop *S);
  ~OpenScop();
  void print(FILE *F);

};

OpenScop::OpenScop(Scop *S) : PollyScop(S) {
  openscop = openscop_scop_malloc();

  initializeArrays();
  initializeParameters();
  initializeScattering();
  initializeStatements();
}

void OpenScop::initializeParameters() {
  openscop->nb_parameters = PollyScop->getNumParams();
  openscop->parameters = new char *[openscop->nb_parameters];

  for (int i = 0; i < openscop->nb_parameters; ++i) {
    openscop->parameters[i] = new char[20];
    sprintf(openscop->parameters[i], "p_%d", i);
  }
}

void OpenScop::initializeArrays() {
  int nb_arrays = 0;

  for (Scop::iterator SI = PollyScop->begin(), SE = PollyScop->end(); SI != SE;
       ++SI)
    for (ScopStmt::memacc_iterator MI = (*SI)->memacc_begin(),
                                   ME = (*SI)->memacc_end();
         MI != ME; ++MI) {
      const Value *BaseAddr = (*MI)->getBaseAddr();
      if (ArrayMap.find(BaseAddr) == ArrayMap.end()) {
        ArrayMap.insert(std::make_pair(BaseAddr, nb_arrays));
        ++nb_arrays;
      }
    }

  openscop->nb_arrays = nb_arrays;
  openscop->arrays = new char *[nb_arrays];

  for (int i = 0; i < nb_arrays; ++i)
    for (std::map<const Value *, int>::iterator VI = ArrayMap.begin(),
                                                VE = ArrayMap.end();
         VI != VE; ++VI)
      if ((*VI).second == i) {
        const Value *V = (*VI).first;
        std::string name = V->getName();
        openscop->arrays[i] = new char[name.size() + 1];
        strcpy(openscop->arrays[i], name.c_str());
      }
}

void OpenScop::initializeScattering() {
  openscop->nb_scattdims = PollyScop->getScatterDim();
  openscop->scattdims = new char *[openscop->nb_scattdims];

  for (int i = 0; i < openscop->nb_scattdims; ++i) {
    openscop->scattdims[i] = new char[20];
    sprintf(openscop->scattdims[i], "s_%d", i);
  }
}

openscop_statement_p OpenScop::initializeStatement(ScopStmt *stmt) {
  openscop_statement_p Stmt = openscop_statement_malloc();

  // Domain & Schedule
  isl_set *domain = stmt->getDomain();
  Stmt->domain = domainToMatrix(domain);
  isl_map *Scattering = stmt->getScattering();
  Stmt->schedule = scatteringToMatrix(Scattering);
  isl_set_free(domain);
  isl_map_free(Scattering);

  // Statement name
  const char *entryName = stmt->getBaseName();
  Stmt->body = (char *)malloc(sizeof(char) * (strlen(entryName) + 1));
  strcpy(Stmt->body, entryName);

  // Iterator names
  Stmt->nb_iterators = isl_set_n_dim(stmt->getDomain());
  Stmt->iterators = new char *[Stmt->nb_iterators];

  for (int i = 0; i < Stmt->nb_iterators; ++i) {
    Stmt->iterators[i] = new char[20];
    sprintf(Stmt->iterators[i], "i_%d", i);
  }

  // Memory Accesses
  Stmt->read = createAccessMatrix(stmt, true);
  Stmt->write = createAccessMatrix(stmt, false);

  return Stmt;
}

void OpenScop::initializeStatements() {
  for (Scop::reverse_iterator SI = PollyScop->rbegin(), SE = PollyScop->rend();
       SI != SE; ++SI) {
    openscop_statement_p stmt = initializeStatement(*SI);
    stmt->next = openscop->statement;
    openscop->statement = stmt;
  }
}

void OpenScop::freeStatement(openscop_statement_p stmt) {

  if (stmt->read)
    openscop_matrix_free(stmt->read);
  stmt->read = NULL;

  if (stmt->write)
    openscop_matrix_free(stmt->write);
  stmt->write = NULL;

  if (stmt->domain)
    openscop_matrix_free(stmt->domain);
  stmt->domain = NULL;

  if (stmt->schedule)
    openscop_matrix_free(stmt->schedule);
  stmt->schedule = NULL;

  for (int i = 0; i < stmt->nb_iterators; ++i)
    delete[](stmt->iterators[i]);

  delete[](stmt->iterators);
  stmt->iterators = NULL;
  stmt->nb_iterators = 0;

  delete[](stmt->body);
  stmt->body = NULL;

  openscop_statement_free(stmt);
}

void OpenScop::print(FILE *F) { openscop_scop_print_dot_scop(F, openscop); }

/// Add an isl constraint to an OpenScop matrix.
///
/// @param user The matrix
/// @param c The constraint
int OpenScop::domainToMatrix_constraint(isl_constraint *c, void *user) {
  openscop_matrix_p m = (openscop_matrix_p) user;

  int nb_params = isl_constraint_dim(c, isl_space_param);
  int nb_vars = isl_constraint_dim(c, isl_space_set);
  int nb_div = isl_constraint_dim(c, isl_space_div);

  assert(!nb_div && "Existentially quantified variables not yet supported");

  openscop_vector_p vec = openscop_vector_malloc(nb_params + nb_vars + 2);

  // Assign type
  if (isl_constraint_is_equality(c))
    openscop_vector_tag_equality(vec);
  else
    openscop_vector_tag_inequality(vec);

  isl_int v;
  isl_int_init(v);

  // Assign variables
  for (int i = 0; i < nb_vars; ++i) {
    isl_constraint_get_coefficient(c, isl_space_set, i, &v);
    isl_int_set(vec->p[i + 1], v);
  }

  // Assign parameters
  for (int i = 0; i < nb_params; ++i) {
    isl_constraint_get_coefficient(c, isl_space_param, i, &v);
    isl_int_set(vec->p[nb_vars + i + 1], v);
  }

  // Assign constant
  isl_constraint_get_constant(c, &v);
  isl_int_set(vec->p[nb_params + nb_vars + 1], v);

  openscop_matrix_insert_vector(m, vec, m->NbRows);

  return 0;
}

/// Add an isl basic set to a OpenScop matrix_list
///
/// @param bset The basic set to add
/// @param user The matrix list we should add the basic set to
///
/// XXX: At the moment this function expects just a matrix, as support
/// for matrix lists is currently not available in OpenScop. So union of
/// polyhedron are not yet supported
int OpenScop::domainToMatrix_basic_set(isl_basic_set *bset, void *user) {
  openscop_matrix_p m = (openscop_matrix_p) user;
  assert(!m->NbRows && "Union of polyhedron not yet supported");

  isl_basic_set_foreach_constraint(bset, &domainToMatrix_constraint, user);
  return 0;
}

/// Translate a isl_set to a OpenScop matrix.
///
/// @param PS The set to be translated
/// @return A OpenScop Matrix
openscop_matrix_p OpenScop::domainToMatrix(isl_set *PS) {

  // Create a canonical copy of this set.
  isl_set *set = isl_set_copy(PS);
  set = isl_set_compute_divs(set);
  set = isl_set_align_divs(set);

  // Initialize the matrix.
  unsigned NbRows, NbColumns;
  NbRows = 0;
  NbColumns = isl_set_n_dim(PS) + isl_set_n_param(PS) + 2;
  openscop_matrix_p matrix = openscop_matrix_malloc(NbRows, NbColumns);

  // Copy the content into the matrix.
  isl_set_foreach_basic_set(set, &domainToMatrix_basic_set, matrix);

  isl_set_free(set);

  return matrix;
}

/// Add an isl constraint to an OpenScop matrix.
///
/// @param user The matrix
/// @param c The constraint
int OpenScop::scatteringToMatrix_constraint(isl_constraint *c, void *user) {
  openscop_matrix_p m = (openscop_matrix_p) user;

  int nb_params = isl_constraint_dim(c, isl_space_param);
  int nb_in = isl_constraint_dim(c, isl_space_in);
  int nb_out = isl_constraint_dim(c, isl_space_out);
  int nb_div = isl_constraint_dim(c, isl_space_div);

  assert(!nb_div && "Existentially quantified variables not yet supported");

  openscop_vector_p vec =
      openscop_vector_malloc(nb_params + nb_in + nb_out + 2);

  // Assign type
  if (isl_constraint_is_equality(c))
    openscop_vector_tag_equality(vec);
  else
    openscop_vector_tag_inequality(vec);

  isl_int v;
  isl_int_init(v);

  // Assign scattering
  for (int i = 0; i < nb_out; ++i) {
    isl_constraint_get_coefficient(c, isl_space_out, i, &v);
    isl_int_set(vec->p[i + 1], v);
  }

  // Assign variables
  for (int i = 0; i < nb_in; ++i) {
    isl_constraint_get_coefficient(c, isl_space_in, i, &v);
    isl_int_set(vec->p[nb_out + i + 1], v);
  }

  // Assign parameters
  for (int i = 0; i < nb_params; ++i) {
    isl_constraint_get_coefficient(c, isl_space_param, i, &v);
    isl_int_set(vec->p[nb_out + nb_in + i + 1], v);
  }

  // Assign constant
  isl_constraint_get_constant(c, &v);
  isl_int_set(vec->p[nb_out + nb_in + nb_params + 1], v);

  openscop_matrix_insert_vector(m, vec, m->NbRows);

  return 0;
}

/// Add an isl basic map to a OpenScop matrix_list
///
/// @param bmap The basic map to add
/// @param user The matrix list we should add the basic map to
///
/// XXX: At the moment this function expects just a matrix, as support
/// for matrix lists is currently not available in OpenScop. So union of
/// polyhedron are not yet supported
int OpenScop::scatteringToMatrix_basic_map(isl_basic_map *bmap, void *user) {
  openscop_matrix_p m = (openscop_matrix_p) user;
  assert(!m->NbRows && "Union of polyhedron not yet supported");

  isl_basic_map_foreach_constraint(bmap, &scatteringToMatrix_constraint, user);
  return 0;
}

/// Translate a isl_map to a OpenScop matrix.
///
/// @param map The map to be translated
/// @return A OpenScop Matrix
openscop_matrix_p OpenScop::scatteringToMatrix(isl_map *pmap) {

  // Create a canonical copy of this set.
  isl_map *map = isl_map_copy(pmap);
  map = isl_map_compute_divs(map);
  map = isl_map_align_divs(map);

  // Initialize the matrix.
  unsigned NbRows, NbColumns;
  NbRows = 0;
  NbColumns =
      isl_map_n_in(pmap) + isl_map_n_out(pmap) + isl_map_n_param(pmap) + 2;
  openscop_matrix_p matrix = openscop_matrix_malloc(NbRows, NbColumns);

  // Copy the content into the matrix.
  isl_map_foreach_basic_map(map, &scatteringToMatrix_basic_map, matrix);

  isl_map_free(map);

  return matrix;
}

/// Add an isl constraint to an OpenScop matrix.
///
/// @param user The matrix
/// @param c The constraint
int OpenScop::accessToMatrix_constraint(isl_constraint *c, void *user) {
  openscop_matrix_p m = (openscop_matrix_p) user;

  int nb_params = isl_constraint_dim(c, isl_space_param);
  int nb_in = isl_constraint_dim(c, isl_space_in);
  int nb_div = isl_constraint_dim(c, isl_space_div);

  assert(!nb_div && "Existentially quantified variables not yet supported");

  openscop_vector_p vec = openscop_vector_malloc(nb_params + nb_in + 2);

  isl_int v;
  isl_int_init(v);

  // The access dimension has to be one.
  isl_constraint_get_coefficient(c, isl_space_out, 0, &v);
  assert(isl_int_is_one(v));
  bool inverse = true;

  // Assign variables
  for (int i = 0; i < nb_in; ++i) {
    isl_constraint_get_coefficient(c, isl_space_in, i, &v);

    if (inverse)
      isl_int_neg(v, v);

    isl_int_set(vec->p[i + 1], v);
  }

  // Assign parameters
  for (int i = 0; i < nb_params; ++i) {
    isl_constraint_get_coefficient(c, isl_space_param, i, &v);

    if (inverse)
      isl_int_neg(v, v);

    isl_int_set(vec->p[nb_in + i + 1], v);
  }

  // Assign constant
  isl_constraint_get_constant(c, &v);

  if (inverse)
    isl_int_neg(v, v);

  isl_int_set(vec->p[nb_in + nb_params + 1], v);

  openscop_matrix_insert_vector(m, vec, m->NbRows);

  return 0;
}

/// Add an isl basic map to a OpenScop matrix_list
///
/// @param bmap The basic map to add
/// @param user The matrix list we should add the basic map to
///
/// XXX: At the moment this function expects just a matrix, as support
/// for matrix lists is currently not available in OpenScop. So union of
/// polyhedron are not yet supported
int OpenScop::accessToMatrix_basic_map(isl_basic_map *bmap, void *user) {
  isl_basic_map_foreach_constraint(bmap, &accessToMatrix_constraint, user);
  return 0;
}

/// Create the memory access matrix for openscop
///
/// @param S The polly statement the access matrix is created for.
/// @param isRead Are we looking for read or write accesses?
/// @param ArrayMap A map translating from the memory references to the openscop
/// indeces
///
/// @return The memory access matrix, as it is required by openscop.
openscop_matrix_p OpenScop::createAccessMatrix(ScopStmt *S, bool isRead) {

  unsigned NbColumns = S->getNumIterators() + S->getNumParams() + 2;
  openscop_matrix_p m = openscop_matrix_malloc(0, NbColumns);

  for (ScopStmt::memacc_iterator MI = S->memacc_begin(), ME = S->memacc_end();
       MI != ME; ++MI)
    if ((*MI)->isRead() == isRead) {
      // Extract the access function.
      isl_map_foreach_basic_map((*MI)->getAccessFunction(),
                                &accessToMatrix_basic_map, m);

      // Set the index of the memory access base element.
      std::map<const Value *, int>::iterator BA =
          ArrayMap.find((*MI)->getBaseAddr());
      isl_int_set_si(m->p[m->NbRows - 1][0], (*BA).second + 1);
    }

  return m;
}

OpenScop::~OpenScop() {
  // Free array names.
  for (int i = 0; i < openscop->nb_arrays; ++i)
    delete[](openscop->arrays[i]);

  delete[](openscop->arrays);
  openscop->arrays = NULL;
  openscop->nb_arrays = 0;

  // Free scattering names.
  for (int i = 0; i < openscop->nb_scattdims; ++i)
    delete[](openscop->scattdims[i]);

  delete[](openscop->scattdims);
  openscop->scattdims = NULL;
  openscop->nb_scattdims = 0;

  // Free parameters
  for (int i = 0; i < openscop->nb_parameters; ++i)
    delete[](openscop->parameters[i]);

  delete[](openscop->parameters);
  openscop->parameters = NULL;
  openscop->nb_parameters = 0;

  openscop_statement_p stmt = openscop->statement;

  // Free Statements
  while (stmt) {
    openscop_statement_p TempStmt = stmt->next;
    stmt->next = NULL;
    freeStatement(stmt);
    stmt = TempStmt;
  }

  openscop->statement = NULL;

  openscop_scop_free(openscop);
}

std::string ScopExporter::getFileName(Scop *S) const {
  std::string FunctionName = S->getRegion().getEntry()->getParent()->getName();
  std::string FileName = FunctionName + "___" + S->getNameStr() + ".scop";
  return FileName;
}

void ScopExporter::printScop(raw_ostream &OS) const { S->print(OS); }

bool ScopExporter::runOnScop(Scop &scop) {
  S = &scop;
  Region &R = S->getRegion();

  std::string FileName = ExportDir + "/" + getFileName(S);
  FILE *F = fopen(FileName.c_str(), "w");

  if (!F) {
    errs() << "Cannot open file: " << FileName << "\n";
    errs() << "Skipping export.\n";
    return false;
  }

  OpenScop openscop(S);
  openscop.print(F);
  fclose(F);

  std::string FunctionName = R.getEntry()->getParent()->getName();
  errs() << "Writing Scop '" << R.getNameStr() << "' in function '"
         << FunctionName << "' to '" << FileName << "'.\n";

  return false;
}

void ScopExporter::getAnalysisUsage(AnalysisUsage &AU) const {
  ScopPass::getAnalysisUsage(AU);
}

static RegisterPass<ScopExporter> A("polly-export",
                                    "Polly - Export Scops with OpenScop library"
                                    " (Writes a .scop file for each Scop)");

Pass *polly::createScopExporterPass() { return new ScopExporter(); }

#endif
