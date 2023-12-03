#include "mlir-c/Dialect/RemainingDialects.h"

#include "mlir/CAPI/Registration.h"
#include "mlir/InitAllDialects.h"

using namespace mlir;

#define MLIR_DEFINE_CAPI_DIALECT_REGISTRATION_(NAMESPACE, NAME)                \
  MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(NAME, NAMESPACE,                       \
                                        NAMESPACE::NAME##Dialect)

#define FORALL_DIALECTS(_)                                                     \
  _(acc, OpenACC)                                                              \
  _(affine, Affine)                                                            \
  _(amx, AMX)                                                                  \
  _(arith, Arith)                                                              \
  _(arm_neon, ArmNeon)                                                         \
  _(arm_sme, ArmSME)                                                           \
  _(arm_sve, ArmSVE)                                                           \
  _(bufferization, Bufferization)                                              \
  _(complex, Complex)                                                          \
  _(emitc, EmitC)                                                              \
  _(index, Index)                                                              \
  _(irdl, IRDL)                                                                \
  _(mesh, Mesh)                                                                \
  _(spirv, SPIRV)                                                              \
  _(tosa, Tosa)                                                                \
  _(ub, UB)                                                                    \
  _(x86vector, X86Vector)

FORALL_DIALECTS(MLIR_DEFINE_CAPI_DIALECT_REGISTRATION_)

#undef MLIR_DEFINE_CAPI_DIALECT_REGISTRATION_
#undef FORALL_DIALECTS

static void mlirDialectRegistryInsertDLTIDialect(MlirDialectRegistry registry) {
  unwrap(registry)->insert<mlir::DLTIDialect>();
}

static MlirDialect mlirContextLoadDLTIDialect(MlirContext context) {
  return wrap(unwrap(context)->getOrLoadDialect<mlir::DLTIDialect>());
}

static MlirStringRef mlirDLTIDialectGetNamespace() {
  return wrap(mlir::DLTIDialect::getDialectNamespace());
}

MlirDialectHandle mlirGetDialectHandle__dlti__() {
  static MlirDialectRegistrationHooks hooks = {
      mlirDialectRegistryInsertDLTIDialect, mlirContextLoadDLTIDialect,
      mlirDLTIDialectGetNamespace};
  return MlirDialectHandle{&hooks};
}
