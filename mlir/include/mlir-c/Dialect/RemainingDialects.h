#ifndef MLIR_C_REMAINING_DIALECTS_H
#define MLIR_C_REMAINING_DIALECTS_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MLIR_DEFINE_CAPI_DIALECT_REGISTRATION_(NAMESPACE)                      \
  MLIR_CAPI_EXPORTED MlirDialectHandle mlirGetDialectHandle__##NAMESPACE##__();

#define FORALL_DIALECTS(_)                                                     \
  _(acc)                                                                       \
  _(affine)                                                                    \
  _(amx)                                                                       \
  _(arm_neon)                                                                  \
  _(arm_sme)                                                                   \
  _(arm_sve)                                                                   \
  _(bufferization)                                                             \
  _(complex)                                                                   \
  _(dlti)                                                                      \
  _(emitc)                                                                     \
  _(index)                                                                     \
  _(irdl)                                                                      \
  _(mesh)                                                                      \
  _(spirv)                                                                     \
  _(tosa)                                                                      \
  _(ub)                                                                        \
  _(x86vector)

FORALL_DIALECTS(MLIR_DEFINE_CAPI_DIALECT_REGISTRATION_)

#undef MLIR_DEFINE_CAPI_DIALECT_REGISTRATION_
#undef FORALL_DIALECTS

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_REMAINING_DIALECTS_H
