//===- ScopLib.h - ScopLib interface ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Scoplib interface.
//
// The scoplib interface allows to import/export a scop using scoplib.
//===----------------------------------------------------------------------===//

#ifndef POLLY_SCOPLIB_H
#define POLLY_SCOPLIB_H

#define SCOPLIB_INT_T_IS_MP
#include "isl/ctx.h"

#include "scoplib/scop.h"

#include <map>

namespace llvm { class Value; }

struct isl_constraint;
struct isl_basic_map;
struct isl_basic_set;
struct isl_map;
struct isl_set;

namespace polly {
class Dependences;
class ScopStmt;
class Scop;
class ScopLib {
  Scop *PollyScop;
  scoplib_scop_p scoplib;
  Dependences *D;

  std::map<const llvm::Value *, int> ArrayMap;

  void initializeArrays();
  void initializeParameters();
  void initializeScattering();
  void initializeStatements();
  scoplib_statement_p initializeStatement(ScopStmt *stmt);
  void freeStatement(scoplib_statement_p stmt);
  static int accessToMatrix_constraint(isl_constraint *c, void *user);
  static int accessToMatrix_basic_map(isl_basic_map *bmap, void *user);
  scoplib_matrix_p createAccessMatrix(ScopStmt *S, bool isRead);
  static int domainToMatrix_constraint(isl_constraint *c, void *user);
  static int domainToMatrix_basic_set(isl_basic_set *bset, void *user);
  scoplib_matrix_p domainToMatrix(__isl_take isl_set *set);
  static int scatteringToMatrix_constraint(isl_constraint *c, void *user);
  static int scatteringToMatrix_basic_map(isl_basic_map *bmap, void *user);
  scoplib_matrix_p scatteringToMatrix(__isl_take isl_map *map);

public:
  ScopLib(Scop *S);
  ScopLib(Scop *S, FILE *F, Dependences *D);
  ~ScopLib();
  void print(FILE *F);
  bool updateScattering();

};
}

#endif /* POLLY_SCOPLIB_H */
