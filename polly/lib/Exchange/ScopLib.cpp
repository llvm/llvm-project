//===- ScopLib.cpp - ScopLib interface ------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// ScopLib Interface
//
//===----------------------------------------------------------------------===//

#include "polly/LinkAllPasses.h"

#ifdef SCOPLIB_FOUND

#include "polly/Dependences.h"
#include "polly/ScopLib.h"
#include "polly/ScopInfo.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Assembly/Writer.h"

#include "stdio.h"
#include "isl/set.h"
#include "isl/map.h"
#include "isl/constraint.h"

using namespace llvm;

namespace polly {

ScopLib::ScopLib(Scop *S) : PollyScop(S) {
  scoplib = scoplib_scop_malloc();

  initializeArrays();
  initializeParameters();
  initializeScattering();
  initializeStatements();
}

ScopLib::ScopLib(Scop *S, FILE *F, Dependences *dep) : PollyScop(S), D(dep) {
  scoplib = scoplib_scop_read(F);
}

void ScopLib::initializeParameters() {
  scoplib->nb_parameters = PollyScop->getNumParams();
  scoplib->parameters = (char**) malloc(sizeof(char*) * scoplib->nb_parameters);

  for (int i = 0; i < scoplib->nb_parameters; ++i) {
    scoplib->parameters[i] = (char *) malloc(sizeof(char*) * 20);
    sprintf(scoplib->parameters[i], "p_%d", i);
  }
}

void ScopLib::initializeArrays() {
  int nb_arrays = 0;

  for (Scop::iterator SI = PollyScop->begin(), SE = PollyScop->end(); SI != SE;
       ++SI)
    for (ScopStmt::memacc_iterator MI = (*SI)->memacc_begin(),
         ME = (*SI)->memacc_end(); MI != ME; ++MI) {
      const Value *BaseAddr = (*MI)->getBaseAddr();
      if (ArrayMap.find(BaseAddr) == ArrayMap.end()) {
        ArrayMap.insert(std::make_pair(BaseAddr, nb_arrays));
        ++nb_arrays;
      }
    }

  scoplib->nb_arrays = nb_arrays;
  scoplib->arrays = (char**)malloc(sizeof(char*) * nb_arrays);

  for (int i = 0; i < nb_arrays; ++i)
    for (std::map<const Value*, int>::iterator VI = ArrayMap.begin(),
         VE = ArrayMap.end(); VI != VE; ++VI)
      if ((*VI).second == i) {
        const Value *V = (*VI).first;
        std::string name = V->getName();
        scoplib->arrays[i] = (char*) malloc(sizeof(char*) * (name.size() + 1));
        strcpy(scoplib->arrays[i], name.c_str());
      }
}

void ScopLib::initializeScattering() {
}

scoplib_statement_p ScopLib::initializeStatement(ScopStmt *stmt) {
  scoplib_statement_p Stmt = scoplib_statement_malloc();

  // Domain & Schedule
  Stmt->domain = scoplib_matrix_list_malloc();
  Stmt->domain->elt = domainToMatrix(stmt->getDomain());
  Stmt->schedule = scatteringToMatrix(stmt->getScattering());

  // Statement name
  std::string entryName;
  raw_string_ostream OS(entryName);
  WriteAsOperand(OS, stmt->getBasicBlock(), false);
  entryName = OS.str();
  Stmt->body = (char*)malloc(sizeof(char) * (entryName.size() + 1));
  strcpy(Stmt->body, entryName.c_str());

  // Iterator names
  Stmt->nb_iterators = stmt->getNumIterators();
  Stmt->iterators = (char**) malloc(sizeof(char*) * Stmt->nb_iterators);

  for (int i = 0; i < Stmt->nb_iterators; ++i) {
    Stmt->iterators[i] = (char*) malloc(sizeof(char*) * 20);
    sprintf(Stmt->iterators[i], "i_%d", i);
  }

  // Memory Accesses
  Stmt->read = createAccessMatrix(stmt, true);
  Stmt->write = createAccessMatrix(stmt, false);

  return Stmt;
}

void ScopLib::initializeStatements() {
  for (Scop::reverse_iterator SI = PollyScop->rbegin(), SE = PollyScop->rend();
       SI != SE; ++SI) {
    scoplib_statement_p stmt = initializeStatement(*SI);
    stmt->next = scoplib->statement;
    scoplib->statement = stmt;
  }
}

void ScopLib::freeStatement(scoplib_statement_p stmt) {

  if (stmt->read)
    scoplib_matrix_free(stmt->read);
  stmt->read = NULL;

  if (stmt->write)
    scoplib_matrix_free(stmt->write);
  stmt->write = NULL;

  scoplib_matrix_list_p current = stmt->domain;
  while (current) {
    scoplib_matrix_list_p next = current->next;
    current->next = NULL;
    scoplib_matrix_free(current->elt);
    current->elt = NULL;
    scoplib_matrix_list_free(current);
    current = next;
  }
  stmt->domain = NULL;

  if (stmt->schedule)
    scoplib_matrix_free(stmt->schedule);
  stmt->schedule = NULL;

  for (int i = 0; i < stmt->nb_iterators; ++i)
    free(stmt->iterators[i]);

  free(stmt->iterators);
  stmt->iterators = NULL;
  stmt->nb_iterators = 0;

  scoplib_statement_free(stmt);
}

void ScopLib::print(FILE *F) {
  scoplib_scop_print_dot_scop(F, scoplib);
}

/// Add an isl constraint to an ScopLib matrix.
///
/// @param user The matrix
/// @param c The constraint
int ScopLib::domainToMatrix_constraint(isl_constraint *c, void *user) {
  scoplib_matrix_p m = (scoplib_matrix_p) user;

  int nb_params = isl_constraint_dim(c, isl_dim_param);
  int nb_vars = isl_constraint_dim(c, isl_dim_set);
  int nb_div = isl_constraint_dim(c, isl_dim_div);

  assert(!nb_div && "Existentially quantified variables not yet supported");

  scoplib_vector_p vec = scoplib_vector_malloc(nb_params + nb_vars + 2);


  // Assign type
  if (isl_constraint_is_equality(c))
    scoplib_vector_tag_equality(vec);
  else
    scoplib_vector_tag_inequality(vec);

  isl_int v;
  isl_int_init(v);

  // Assign variables
  for (int i = 0; i < nb_vars; ++i) {
    isl_constraint_get_coefficient(c, isl_dim_set, i, &v);
    isl_int_set(vec->p[i + 1], v);
  }

  // Assign parameters
  for (int i = 0; i < nb_params; ++i) {
    isl_constraint_get_coefficient(c, isl_dim_param, i, &v);
    isl_int_set(vec->p[nb_vars + i + 1], v);
  }

  // Assign constant
  isl_constraint_get_constant(c, &v);
  isl_int_set(vec->p[nb_params + nb_vars + 1], v);

  scoplib_matrix_insert_vector(m, vec, m->NbRows);

  scoplib_vector_free(vec);
  isl_constraint_free(c);
  isl_int_clear(v);

  return 0;
}

/// Add an isl basic set to a ScopLib matrix_list
///
/// @param bset The basic set to add
/// @param user The matrix list we should add the basic set to
///
/// XXX: At the moment this function expects just a matrix, as support
/// for matrix lists is currently not available in ScopLib. So union of
/// polyhedron are not yet supported
int ScopLib::domainToMatrix_basic_set(isl_basic_set *bset, void *user) {
  scoplib_matrix_p m = (scoplib_matrix_p) user;
  assert(!m->NbRows && "Union of polyhedron not yet supported");

  isl_basic_set_foreach_constraint(bset, &domainToMatrix_constraint, user);
  isl_basic_set_free(bset);
  return 0;
}

/// Translate a isl_set to a ScopLib matrix.
///
/// @param PS The set to be translated
/// @return A ScopLib Matrix
scoplib_matrix_p ScopLib::domainToMatrix(__isl_take isl_set *set) {
  set = isl_set_compute_divs (set);
  set = isl_set_align_divs (set);

  // Initialize the matrix.
  unsigned NbRows, NbColumns;
  NbRows = 0;
  NbColumns = isl_set_n_dim(set) + isl_set_n_param(set) + 2;
  scoplib_matrix_p matrix = scoplib_matrix_malloc(NbRows, NbColumns);

  // Copy the content into the matrix.
  isl_set_foreach_basic_set(set, &domainToMatrix_basic_set, matrix);

  isl_set_free(set);

  return matrix;
}

/// Add an isl constraint to an ScopLib matrix.
///
/// @param user The matrix
/// @param c The constraint
int ScopLib::scatteringToMatrix_constraint(isl_constraint *c, void *user) {
  scoplib_matrix_p m = (scoplib_matrix_p) user;

  int nb_params = isl_constraint_dim(c, isl_dim_param);
  int nb_in = isl_constraint_dim(c, isl_dim_in);
  int nb_div = isl_constraint_dim(c, isl_dim_div);

  assert(!nb_div && "Existentially quantified variables not yet supported");

  scoplib_vector_p vec =
    scoplib_vector_malloc(nb_params + nb_in + 2);

  // Assign type
  if (isl_constraint_is_equality(c))
    scoplib_vector_tag_equality(vec);
  else
    scoplib_vector_tag_inequality(vec);

  isl_int v;
  isl_int_init(v);

  // Assign variables
  for (int i = 0; i < nb_in; ++i) {
    isl_constraint_get_coefficient(c, isl_dim_in, i, &v);
    isl_int_set(vec->p[i + 1], v);
  }

  // Assign parameters
  for (int i = 0; i < nb_params; ++i) {
    isl_constraint_get_coefficient(c, isl_dim_param, i, &v);
    isl_int_set(vec->p[nb_in + i + 1], v);
  }

  // Assign constant
  isl_constraint_get_constant(c, &v);
  isl_int_set(vec->p[nb_in + nb_params + 1], v);

  scoplib_vector_p null =
    scoplib_vector_malloc(nb_params + nb_in + 2);

  vec = scoplib_vector_sub(null, vec);
  scoplib_matrix_insert_vector(m, vec, 0);

  isl_constraint_free(c);
  isl_int_clear(v);

  return 0;
}

/// Add an isl basic map to a ScopLib matrix_list
///
/// @param bmap The basic map to add
/// @param user The matrix list we should add the basic map to
///
/// XXX: At the moment this function expects just a matrix, as support
/// for matrix lists is currently not available in ScopLib. So union of
/// polyhedron are not yet supported
int ScopLib::scatteringToMatrix_basic_map(isl_basic_map *bmap, void *user) {
  scoplib_matrix_p m = (scoplib_matrix_p) user;
  assert(!m->NbRows && "Union of polyhedron not yet supported");

  isl_basic_map_foreach_constraint(bmap, &scatteringToMatrix_constraint, user);
  isl_basic_map_free(bmap);
  return 0;
}

/// Translate a isl_map to a ScopLib matrix.
///
/// @param map The map to be translated
/// @return A ScopLib Matrix
scoplib_matrix_p ScopLib::scatteringToMatrix(__isl_take isl_map *map) {
  map = isl_map_compute_divs (map);
  map = isl_map_align_divs (map);

  // Initialize the matrix.
  unsigned NbRows, NbColumns;
  NbRows = 0;
  NbColumns = isl_map_n_in(map) + isl_map_n_param(map) + 2;
  scoplib_matrix_p matrix = scoplib_matrix_malloc(NbRows, NbColumns);

  // Copy the content into the matrix.
  isl_map_foreach_basic_map(map, &scatteringToMatrix_basic_map, matrix);

  // Only keep the relevant rows.
  scoplib_matrix_p reduced = scoplib_matrix_ncopy(matrix,
                                                  isl_map_n_in(map) * 2 + 1);

  scoplib_matrix_free (matrix);
  isl_map_free(map);

  return reduced;
}

/// Add an isl constraint to an ScopLib matrix.
///
/// @param user The matrix
/// @param c The constraint
int ScopLib::accessToMatrix_constraint(isl_constraint *c, void *user) {
  scoplib_matrix_p m = (scoplib_matrix_p) user;

  int nb_params = isl_constraint_dim(c, isl_dim_param);
  int nb_in = isl_constraint_dim(c, isl_dim_in);
  int nb_div = isl_constraint_dim(c, isl_dim_div);

  assert(!nb_div && "Existentially quantified variables not yet supported");

  scoplib_vector_p vec =
    scoplib_vector_malloc(nb_params + nb_in + 2);

  isl_int v;
  isl_int_init(v);

  // The access dimension has to be one.
  isl_constraint_get_coefficient(c, isl_dim_out, 0, &v);
  assert((isl_int_is_one(v) || isl_int_is_negone(v))
         && "Access relations not supported in scoplib");
  bool inverse = isl_int_is_one(v);

  // Assign variables
  for (int i = 0; i < nb_in; ++i) {
    isl_constraint_get_coefficient(c, isl_dim_in, i, &v);

    if (inverse) isl_int_neg(v,v);

    isl_int_set(vec->p[i + 1], v);
  }

  // Assign parameters
  for (int i = 0; i < nb_params; ++i) {
    isl_constraint_get_coefficient(c, isl_dim_param, i, &v);

    if (inverse) isl_int_neg(v,v);

    isl_int_set(vec->p[nb_in + i + 1], v);
  }

  // Assign constant
  isl_constraint_get_constant(c, &v);

  if (inverse) isl_int_neg(v,v);

  isl_int_set(vec->p[nb_in + nb_params + 1], v);

  scoplib_matrix_insert_vector(m, vec, m->NbRows);

  isl_constraint_free(c);
  isl_int_clear(v);

  return 0;
}


/// Add an isl basic map to a ScopLib matrix_list
///
/// @param bmap The basic map to add
/// @param user The matrix list we should add the basic map to
///
/// XXX: At the moment this function expects just a matrix, as support
/// for matrix lists is currently not available in ScopLib. So union of
/// polyhedron are not yet supported
int ScopLib::accessToMatrix_basic_map(isl_basic_map *bmap, void *user) {
  isl_basic_map_foreach_constraint(bmap, &accessToMatrix_constraint, user);
  isl_basic_map_free(bmap);
  return 0;
}

/// Create the memory access matrix for scoplib
///
/// @param S The polly statement the access matrix is created for.
/// @param isRead Are we looking for read or write accesses?
/// @param ArrayMap A map translating from the memory references to the scoplib
/// indeces
///
/// @return The memory access matrix, as it is required by scoplib.
scoplib_matrix_p ScopLib::createAccessMatrix(ScopStmt *S, bool isRead) {

  unsigned NbColumns = S->getNumIterators() + S->getNumParams() + 2;
  scoplib_matrix_p m = scoplib_matrix_malloc(0, NbColumns);

  for (ScopStmt::memacc_iterator MI = S->memacc_begin(), ME = S->memacc_end();
       MI != ME; ++MI)
    if ((*MI)->isRead() == isRead) {
      // Extract the access function.
      isl_map *AccessRelation = (*MI)->getAccessRelation();
      isl_map_foreach_basic_map(AccessRelation,
                                &accessToMatrix_basic_map, m);
      isl_map_free(AccessRelation);

      // Set the index of the memory access base element.
      std::map<const Value*, int>::iterator BA =
        ArrayMap.find((*MI)->getBaseAddr());
      isl_int_set_si(m->p[m->NbRows - 1][0], (*BA).second + 1);
    }

  return m;
}

ScopLib::~ScopLib() {
  if (!scoplib)
    return;

  // Free array names.
  for (int i = 0; i < scoplib->nb_arrays; ++i)
    free(scoplib->arrays[i]);

  free(scoplib->arrays);
  scoplib->arrays = NULL;
  scoplib->nb_arrays = 0;

  // Free parameters
  for (int i = 0; i < scoplib->nb_parameters; ++i)
    free(scoplib->parameters[i]);

  free(scoplib->parameters);
  scoplib->parameters = NULL;
  scoplib->nb_parameters = 0;

  scoplib_statement_p stmt = scoplib->statement;

  // Free Statements
  while (stmt) {
    scoplib_statement_p TempStmt = stmt->next;
    stmt->next = NULL;
    freeStatement(stmt);
    stmt = TempStmt;
  }

  scoplib->statement = NULL;

  scoplib_scop_free(scoplib);
}
/// @brief Create an isl constraint from a row of OpenScop integers.
///
/// @param row An array of isl/OpenScop integers.
/// @param Space An isl space object, describing how to spilt the dimensions.
///
/// @return An isl constraint representing this integer array.
isl_constraint *constraintFromMatrixRow(isl_int *row,
                                        __isl_take isl_space *Space) {
  isl_constraint *c;

  unsigned NbIn = isl_space_dim(Space, isl_dim_in);
  unsigned NbParam = isl_space_dim(Space, isl_dim_param);

  if (isl_int_is_zero(row[0]))
    c = isl_equality_alloc(isl_local_space_from_space(Space));
  else
    c = isl_inequality_alloc(isl_local_space_from_space(Space));

  unsigned current_column = 1;

  for (unsigned j = 0; j < NbIn; ++j)
    isl_constraint_set_coefficient(c, isl_dim_in, j, row[current_column++]);

  for (unsigned j = 0; j < NbParam; ++j)
    isl_constraint_set_coefficient(c, isl_dim_param, j, row[current_column++]);

  isl_constraint_set_constant(c, row[current_column]);

  return c;
}

/// @brief Create an isl map from a OpenScop matrix.
///
/// @param m The OpenScop matrix to translate.
/// @param Space The dimensions that are contained in the OpenScop matrix.
///
/// @return An isl map representing m.
isl_map *mapFromMatrix(scoplib_matrix_p m, __isl_take isl_space *Space,
                       unsigned scatteringDims) {
  isl_basic_map *bmap = isl_basic_map_universe(isl_space_copy(Space));

  for (unsigned i = 0; i < m->NbRows; ++i) {
    isl_constraint *c;

    c = constraintFromMatrixRow(m->p[i], isl_space_copy(Space));

    mpz_t minusOne;
    mpz_init(minusOne);
    mpz_set_si(minusOne, -1);
    isl_constraint_set_coefficient(c, isl_dim_out, i, minusOne);

    bmap = isl_basic_map_add_constraint(bmap, c);
  }

  for (unsigned i = m->NbRows; i < scatteringDims; i++) {
    isl_constraint *c;

    c = isl_equality_alloc(isl_local_space_from_space(isl_space_copy(Space)));

    mpz_t One;
    mpz_init(One);
    mpz_set_si(One, 1);
    isl_constraint_set_coefficient(c, isl_dim_out, i, One);

    bmap = isl_basic_map_add_constraint(bmap, c);
  }

  isl_space_free(Space);

  return isl_map_from_basic_map(bmap);
}
/// @brief Create an isl constraint from a row of OpenScop integers.
///
/// @param row An array of isl/OpenScop integers.
/// @param Space An isl space object, describing how to spilt the dimensions.
///
/// @return An isl constraint representing this integer array.
isl_constraint *constraintFromMatrixRowFull(isl_int *row,
                                            __isl_take isl_space *Space) {
  isl_constraint *c;

  unsigned NbOut = isl_space_dim(Space, isl_dim_out);
  unsigned NbIn = isl_space_dim(Space, isl_dim_in);
  unsigned NbParam = isl_space_dim(Space, isl_dim_param);

  isl_local_space *LSpace = isl_local_space_from_space(Space);

  if (isl_int_is_zero(row[0]))
    c = isl_equality_alloc(LSpace);
  else
    c = isl_inequality_alloc(LSpace);

  unsigned current_column = 1;

  for (unsigned j = 0; j < NbOut; ++j)
    isl_constraint_set_coefficient(c, isl_dim_out, j, row[current_column++]);

  for (unsigned j = 0; j < NbIn; ++j)
    isl_constraint_set_coefficient(c, isl_dim_in, j, row[current_column++]);

  for (unsigned j = 0; j < NbParam; ++j)
    isl_constraint_set_coefficient(c, isl_dim_param, j, row[current_column++]);

  isl_constraint_set_constant(c, row[current_column]);

  return c;
}

/// @brief Create an isl map from a OpenScop matrix.
///
/// @param m The OpenScop matrix to translate.
/// @param Space The dimensions that are contained in the OpenScop matrix.
///
/// @return An isl map representing m.
isl_map *mapFromMatrix(scoplib_matrix_p m, __isl_take isl_space *Space) {
  isl_basic_map *bmap = isl_basic_map_universe(isl_space_copy(Space));

  for (unsigned i = 0; i < m->NbRows; ++i) {
    isl_constraint *c;

    c = constraintFromMatrixRowFull(m->p[i], isl_space_copy(Space));
    bmap = isl_basic_map_add_constraint(bmap, c);
  }

  isl_space_free(Space);

  return isl_map_from_basic_map(bmap);
}

/// @brief Create a new scattering for PollyStmt.
///
/// @param m The matrix describing the new scattering.
/// @param PollyStmt The statement to create the scattering for.
///
/// @return An isl_map describing the scattering.
isl_map *scatteringForStmt(scoplib_matrix_p m, ScopStmt *PollyStmt,
                           int scatteringDims) {

  unsigned NbParam = PollyStmt->getNumParams();
  unsigned NbIterators = PollyStmt->getNumIterators();
  unsigned NbScattering;

  if (scatteringDims == -1)
    NbScattering = m->NbColumns - 2 - NbParam - NbIterators;
  else
    NbScattering = scatteringDims;

  isl_ctx *ctx = PollyStmt->getParent()->getIslCtx();
  isl_space *Space = isl_dim_alloc(ctx, NbParam, NbIterators, NbScattering);

  isl_space *ParamSpace = PollyStmt->getParent()->getParamSpace();

  // We need to copy the isl_ids for the parameter dimensions to the new
  // map. Without doing this the current map would have different
  // ids then the new one, even though both are named identically.
  for (unsigned i = 0; i < isl_space_dim(Space, isl_dim_param);
       i++) {
    isl_id *id = isl_space_get_dim_id(ParamSpace, isl_dim_param, i);
    Space = isl_space_set_dim_id(Space, isl_dim_param, i, id);
  }

  isl_space_free(ParamSpace);

  Space = isl_space_set_tuple_name(Space, isl_dim_out, "scattering");
  Space = isl_space_set_tuple_id(Space, isl_dim_in, PollyStmt->getDomainId());

  if (scatteringDims == -1)
    return mapFromMatrix(m, Space);

  return mapFromMatrix(m, Space, scatteringDims);
}

unsigned maxScattering(scoplib_statement_p stmt) {
  unsigned max = 0;

  while (stmt) {
    max = std::max(max, stmt->schedule->NbRows);
    stmt = stmt->next;
  }

  return max;
}

typedef Dependences::StatementToIslMapTy StatementToIslMapTy;

void freeStmtToIslMap(StatementToIslMapTy *Map) {
  for (StatementToIslMapTy::iterator MI = Map->begin(), ME = Map->end();
       MI != ME; ++MI)
    isl_map_free(MI->second);

  delete (Map);
}

/// @brief Read the new scattering from the scoplib description.
///
/// @S      The Scop to update
/// @OScop  The ScopLib data structure describing the new scattering.
/// @return A map that contains for each Statement the new scattering.
StatementToIslMapTy *readScattering(Scop *S, scoplib_scop_p OScop) {
  StatementToIslMapTy &NewScattering = *(new StatementToIslMapTy());

  scoplib_statement_p stmt = OScop->statement;

  // Check if we have dimensions for each scattering or if each row
  // represents a scattering dimension.
  int numScatteringDims = -1;
  ScopStmt *pollyStmt = *S->begin();

  if (stmt->schedule->NbColumns
      == 2 + pollyStmt->getNumParams() + pollyStmt->getNumIterators()) {
    numScatteringDims = maxScattering(stmt);
  }

  for (Scop::iterator SI = S->begin(), SE = S->end(); SI != SE; ++SI) {
    if (!stmt) {
      errs() << "Not enough statements available in OpenScop file\n";
      freeStmtToIslMap(&NewScattering);
      return NULL;
    }

    NewScattering[*SI] = scatteringForStmt(stmt->schedule, *SI,
                                           numScatteringDims);
    stmt = stmt->next;
  }

  if (stmt) {
    errs() << "Too many statements in OpenScop file\n";
    freeStmtToIslMap(&NewScattering);
    return NULL;
  }

  return &NewScattering;
}

/// @brief Update the scattering in a Scop using the scoplib description of
/// the scattering.
bool ScopLib::updateScattering() {
  if (!scoplib)
    return false;

  StatementToIslMapTy *NewScattering = readScattering(PollyScop, scoplib);

  if (!NewScattering)
    return false;

  if (!D->isValidScattering(NewScattering)) {
    freeStmtToIslMap(NewScattering);
    errs() << "OpenScop file contains a scattering that changes the "
      << "dependences. Use -disable-polly-legality to continue anyways\n";
    return false;
  }

  for (Scop::iterator SI = PollyScop->begin(), SE = PollyScop->end(); SI != SE;
       ++SI) {
    ScopStmt *Stmt = *SI;

    if (NewScattering->find(Stmt) != NewScattering->end())
      Stmt->setScattering(isl_map_copy((*NewScattering)[Stmt]));
  }

  freeStmtToIslMap(NewScattering);
  return true;
}
}

#endif
