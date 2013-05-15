//===------ polly/CodeGeneration.h - The Polly code generator *- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_CODEGENERATION_H
#define POLLY_CODEGENERATION_H

#include "polly/Config/config.h"

#include "isl/set.h"
#include "isl/map.h"

namespace polly {
enum VectorizerChoice {
  VECTORIZER_NONE,
  VECTORIZER_POLLY,
  VECTORIZER_UNROLL_ONLY,
  VECTORIZER_FIRST_NEED_GROUPED_UNROLL = VECTORIZER_UNROLL_ONLY,
  VECTORIZER_BB
};
extern VectorizerChoice PollyVectorizerChoice;

static inline int getNumberOfIterations(__isl_take isl_set *Domain) {
  int Dim = isl_set_dim(Domain, isl_dim_set);

  // Calculate a map similar to the identity map, but with the last input
  // and output dimension not related.
  //  [i0, i1, i2, i3] -> [i0, i1, i2, o0]
  isl_space *Space = isl_set_get_space(Domain);
  Space = isl_space_drop_dims(Space, isl_dim_out, Dim - 1, 1);
  Space = isl_space_map_from_set(Space);
  isl_map *Identity = isl_map_identity(Space);
  Identity = isl_map_add_dims(Identity, isl_dim_in, 1);
  Identity = isl_map_add_dims(Identity, isl_dim_out, 1);

  isl_map *Map =
      isl_map_from_domain_and_range(isl_set_copy(Domain), isl_set_copy(Domain));
  isl_set_free(Domain);
  Map = isl_map_intersect(Map, Identity);

  isl_map *LexMax = isl_map_lexmax(isl_map_copy(Map));
  isl_map *LexMin = isl_map_lexmin(Map);
  isl_map *Sub = isl_map_sum(LexMax, isl_map_neg(LexMin));

  isl_set *Elements = isl_map_range(Sub);

  if (!isl_set_is_singleton(Elements)) {
    isl_set_free(Elements);
    return -1;
  }

  isl_point *P = isl_set_sample_point(Elements);

  isl_int V;
  isl_int_init(V);
  isl_point_get_coordinate(P, isl_dim_set, Dim - 1, &V);
  int NumberIterations = isl_int_get_si(V);
  isl_int_clear(V);
  isl_point_free(P);

  return NumberIterations;
}

}

#endif // POLLY_CODEGENERATION_H
