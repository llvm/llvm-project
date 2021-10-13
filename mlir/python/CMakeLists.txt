include(AddMLIRPython)

################################################################################
# Structural groupings.
################################################################################

declare_mlir_python_sources(MLIRPythonSources)
declare_mlir_python_sources(MLIRPythonSources.Dialects
  ADD_TO_PARENT MLIRPythonSources)

declare_mlir_python_sources(MLIRPythonTestSources)
declare_mlir_python_sources(MLIRPythonTestSources.Dialects
  ADD_TO_PARENT MLIRPythonTestSources)

################################################################################
# Pure python sources and generated code
################################################################################

declare_mlir_python_sources(MLIRPythonSources.Core
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  ADD_TO_PARENT MLIRPythonSources
  SOURCES
    _mlir_libs/__init__.py
    ir.py
    passmanager.py
    dialects/_ods_common.py
)

declare_mlir_python_sources(MLIRPythonSources.ExecutionEngine
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  ADD_TO_PARENT MLIRPythonSources
  SOURCES
    execution_engine.py
  SOURCES_GLOB
    runtime/*.py
)

declare_mlir_python_sources(MLIRPythonSources.Passes
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  ADD_TO_PARENT MLIRPythonSources
  SOURCES_GLOB
    all_passes_registration/*.py
    conversions/*.py
    transforms/*.py
)

declare_mlir_python_sources(MLIRPythonCAPIHeaderSources
  ROOT_DIR "${MLIR_SOURCE_DIR}/include"
  SOURCES_GLOB "mlir-c/*.h"
  DEST_PREFIX "_mlir_libs/include"
)

################################################################################
# Dialect bindings
################################################################################

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT MLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  TD_FILE dialects/AsyncOps.td
  SOURCES_GLOB dialects/async_dialect/*.py
  DIALECT_NAME async_dialect)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT MLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  TD_FILE dialects/BuiltinOps.td
  SOURCES
    dialects/builtin.py
    dialects/_builtin_ops_ext.py
  DIALECT_NAME builtin)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT MLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  TD_FILE dialects/GPUOps.td
  SOURCES_GLOB dialects/gpu/*.py
  DIALECT_NAME gpu)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT MLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  TD_FILE dialects/LinalgOps.td
  SOURCES
    dialects/_linalg_ops_ext.py
  SOURCES_GLOB
    dialects/linalg/*.py
  DIALECT_NAME linalg
  DEPENDS LinalgOdsGen)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT MLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  TD_FILE dialects/MathOps.td
  SOURCES dialects/math.py
  DIALECT_NAME math)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT MLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  TD_FILE dialects/ArithmeticOps.td
  SOURCES
    dialects/arith.py
    dialects/_arith_ops_ext.py
  DIALECT_NAME arith)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT MLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  TD_FILE dialects/MemRefOps.td
  SOURCES dialects/memref.py
  DIALECT_NAME memref)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT MLIRPythonTestSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  TD_FILE dialects/PythonTest.td
  SOURCES dialects/python_test.py
  DIALECT_NAME python_test)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT MLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  TD_FILE dialects/SCFOps.td
  SOURCES
    dialects/scf.py
    dialects/_scf_ops_ext.py
  DIALECT_NAME scf)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT MLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  TD_FILE dialects/ShapeOps.td
  SOURCES dialects/shape.py
  DIALECT_NAME shape)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT MLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  TD_FILE dialects/SparseTensorOps.td
  SOURCES dialects/sparse_tensor.py
  DIALECT_NAME sparse_tensor)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT MLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  TD_FILE dialects/StandardOps.td
  SOURCES
    dialects/std.py
    dialects/_std_ops_ext.py
  DIALECT_NAME std)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT MLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  TD_FILE dialects/TensorOps.td
  SOURCES dialects/tensor.py
  DIALECT_NAME tensor)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT MLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  TD_FILE dialects/TosaOps.td
  SOURCES dialects/tosa.py
  DIALECT_NAME tosa)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT MLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  TD_FILE dialects/VectorOps.td
  SOURCES dialects/vector.py
  DIALECT_NAME vector)

################################################################################
# Python extensions.
# The sources for these are all in lib/Bindings/Python, but since they have to
# be rebuilt for each package and integrate with the source setup here, we
# just reference them here instead of having ordered, cross package target
# dependencies.
################################################################################

set(PYTHON_SOURCE_DIR "${MLIR_SOURCE_DIR}/lib/Bindings/Python")
declare_mlir_python_extension(MLIRPythonExtension.Core
  MODULE_NAME _mlir
  ADD_TO_PARENT MLIRPythonSources.Core
  SOURCES
    ${PYTHON_SOURCE_DIR}/DialectLinalg.cpp  # TODO: Break this out.
    ${PYTHON_SOURCE_DIR}/DialectSparseTensor.cpp  # TODO: Break this out.
    ${PYTHON_SOURCE_DIR}/MainModule.cpp
    ${PYTHON_SOURCE_DIR}/IRAffine.cpp
    ${PYTHON_SOURCE_DIR}/IRAttributes.cpp
    ${PYTHON_SOURCE_DIR}/IRCore.cpp
    ${PYTHON_SOURCE_DIR}/IRModule.cpp
    ${PYTHON_SOURCE_DIR}/IRTypes.cpp
    ${PYTHON_SOURCE_DIR}/PybindUtils.cpp
    ${PYTHON_SOURCE_DIR}/Pass.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIDebug
    MLIRCAPIIR
    MLIRCAPIRegistration  # TODO: See about dis-aggregating

    # Dialects
    MLIRCAPILinalg  # TODO: Remove when above is removed.
    MLIRCAPISparseTensor  # TODO: Remove when above is removed.
    MLIRCAPIStandard
)

declare_mlir_python_extension(MLIRPythonExtension.AllPassesRegistration
  MODULE_NAME _mlirAllPassesRegistration
  SOURCES
    ${PYTHON_SOURCE_DIR}/AllPassesRegistration.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIConversion
    MLIRCAPITransforms
)

declare_mlir_python_extension(MLIRPythonExtension.AsyncDialectPasses
  MODULE_NAME _mlirAsyncPasses
  ADD_TO_PARENT MLIRPythonSources.Dialects.async_dialect
  SOURCES
    ${PYTHON_SOURCE_DIR}/AsyncPasses.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIAsync
)

declare_mlir_python_extension(MLIRPythonExtension.Conversions
  MODULE_NAME _mlirConversions
  ADD_TO_PARENT MLIRPythonSources.Passes
  SOURCES
    ${PYTHON_SOURCE_DIR}/Conversions/Conversions.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
  MLIRCAPIConversion
)

declare_mlir_python_extension(MLIRPythonExtension.ExecutionEngine
  MODULE_NAME _mlirExecutionEngine
  ADD_TO_PARENT MLIRPythonSources.ExecutionEngine
  SOURCES
    ${PYTHON_SOURCE_DIR}/ExecutionEngineModule.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    MLIRCEXECUTIONENGINE
)

declare_mlir_python_extension(MLIRPythonExtension.GPUDialectPasses
  MODULE_NAME _mlirGPUPasses
  ADD_TO_PARENT MLIRPythonSources.Dialects.gpu
  SOURCES
    ${PYTHON_SOURCE_DIR}/GPUPasses.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIGPU
)

declare_mlir_python_extension(MLIRPythonExtension.LinalgPasses
  MODULE_NAME _mlirLinalgPasses
  ADD_TO_PARENT MLIRPythonSources.Dialects.linalg
  SOURCES
    ${PYTHON_SOURCE_DIR}/LinalgPasses.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    MLIRCAPILinalg
)

declare_mlir_python_extension(MLIRPythonExtension.SparseTensorDialectPasses
  MODULE_NAME _mlirSparseTensorPasses
  ADD_TO_PARENT MLIRPythonSources.Dialects.sparse_tensor
  SOURCES
    ${PYTHON_SOURCE_DIR}/SparseTensorPasses.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    MLIRCAPISparseTensor
)

declare_mlir_python_extension(MLIRPythonExtension.Transforms
  MODULE_NAME _mlirTransforms
  ADD_TO_PARENT MLIRPythonSources.Passes
  SOURCES
    ${PYTHON_SOURCE_DIR}/Transforms/Transforms.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    MLIRCAPITransforms
)

################################################################################
# Common CAPI dependency DSO.
# All python extensions must link through one DSO which exports the CAPI, and
# this must have a globally unique name amongst all embeddors of the python
# library since it will effectively have global scope.
#
# The presence of this aggregate library is part of the long term plan, but its
# use needs to be made more flexible.
#
# TODO: Upgrade to the aggregate utility in https://reviews.llvm.org/D106419
# once ready.
################################################################################

add_mlir_python_common_capi_library(MLIRPythonCAPI
  INSTALL_COMPONENT MLIRPythonModules
  INSTALL_DESTINATION python_packages/mlir_core/mlir/_mlir_libs
  OUTPUT_DIRECTORY "${MLIR_BINARY_DIR}/python_packages/mlir_core/mlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    MLIRPythonSources
    MLIRPythonExtension.AllPassesRegistration
)

################################################################################
# The fully assembled package of modules.
# This must come last.
################################################################################

add_mlir_python_modules(MLIRPythonModules
  ROOT_PREFIX "${MLIR_BINARY_DIR}/python_packages/mlir_core/mlir"
  INSTALL_PREFIX "python_packages/mlir_core/mlir"
  DECLARED_SOURCES
    MLIRPythonSources
    MLIRPythonExtension.AllPassesRegistration
    MLIRPythonCAPIHeaderSources
  COMMON_CAPI_LINK_LIBS
    MLIRPythonCAPI
  )


add_mlir_python_modules(MLIRPythonTestModules
  ROOT_PREFIX "${MLIR_BINARY_DIR}/python_packages/mlir_test/mlir"
  INSTALL_PREFIX "python_packages/mlir_test/mlir"
  DECLARED_SOURCES
    MLIRPythonTestSources
  )
