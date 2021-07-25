include(AddMLIRPython)
add_custom_target(MLIRBindingsPythonExtension)

################################################################################
# All python extensions must link through one DSO which exports the CAPI, and
# this must have a globally unique name amongst all embeddors of the python
# library since it will effectively have global scope.
#
# The presence of this aggregate library is part of the long term plan, but its
# use needs to be made more flexible.
################################################################################

set(public_api_libs
  MLIRCAPIConversion
  MLIRCAPIDebug
  MLIRCEXECUTIONENGINE
  MLIRCAPIIR
  MLIRCAPIRegistration
  MLIRCAPITransforms

  # Dialects
  MLIRCAPIAsync
  MLIRCAPIGPU
  MLIRCAPILinalg
  MLIRCAPILLVM
  MLIRCAPIShape
  MLIRCAPISparseTensor
  MLIRCAPIStandard
  MLIRCAPISCF
  MLIRCAPITensor
)

foreach(lib ${public_api_libs})
  if(XCODE)
    # Xcode doesn't support object libraries, so we have to trick it into
    # linking the static libraries instead.
    list(APPEND _DEPS "-force_load" ${lib})
  else()
    list(APPEND _OBJECTS $<TARGET_OBJECTS:obj.${lib}>)
  endif()
  # Accumulate transitive deps of each exported lib into _DEPS.
  list(APPEND _DEPS  $<TARGET_PROPERTY:${lib},LINK_LIBRARIES>)
endforeach()

add_mlir_library(MLIRPythonCAPI
  PARTIAL_SOURCES_INTENDED
  SHARED
  ${_OBJECTS}
  EXCLUDE_FROM_LIBMLIR
  LINK_LIBS
  ${_DEPS}
)
if(MSVC)
  set_property(TARGET MLIRPythonCAPI PROPERTY WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

################################################################################
# Build core python extension
################################################################################
add_mlir_python_extension(MLIRCoreBindingsPythonExtension _mlir
  INSTALL_DIR
    python
  SOURCES
    DialectLinalg.cpp
    DialectSparseTensor.cpp
    MainModule.cpp
    IRAffine.cpp
    IRAttributes.cpp
    IRCore.cpp
    IRModule.cpp
    IRTypes.cpp
    PybindUtils.cpp
    Pass.cpp
    ExecutionEngine.cpp
  LINK_LIBS PRIVATE
    LLVMSupport
    MLIRPythonCAPI
)
add_dependencies(MLIRBindingsPythonExtension MLIRCoreBindingsPythonExtension)

add_subdirectory(Transforms)
add_subdirectory(Conversions)

add_mlir_python_extension(MLIRAllPassesRegistrationBindingsPythonExtension _mlirAllPassesRegistration
  INSTALL_DIR
    python
  SOURCES
    AllPassesRegistration.cpp
  LINK_LIBS PRIVATE
    LLVMSupport
    MLIRPythonCAPI
)
add_dependencies(MLIRBindingsPythonExtension MLIRAllPassesRegistrationBindingsPythonExtension)

add_mlir_python_extension(MLIRAsyncPassesBindingsPythonExtension _mlirAsyncPasses
  INSTALL_DIR
    python
  SOURCES
    AsyncPasses.cpp
  LINK_LIBS PRIVATE
    LLVMSupport
    MLIRPythonCAPI
)
add_dependencies(MLIRBindingsPythonExtension MLIRAsyncPassesBindingsPythonExtension)

add_mlir_python_extension(MLIRSparseTensorPassesBindingsPythonExtension _mlirSparseTensorPasses
  INSTALL_DIR
    python
  SOURCES
    SparseTensorPasses.cpp
  LINK_LIBS PRIVATE
    LLVMSupport
    MLIRPythonCAPI
)
add_dependencies(MLIRBindingsPythonExtension MLIRSparseTensorPassesBindingsPythonExtension)

add_mlir_python_extension(MLIRGPUPassesBindingsPythonExtension _mlirGPUPasses
  INSTALL_DIR
    python
  SOURCES
    GPUPasses.cpp
  LINK_LIBS PRIVATE
    LLVMSupport
    MLIRPythonCAPI
)
add_dependencies(MLIRBindingsPythonExtension MLIRGPUPassesBindingsPythonExtension)

add_mlir_python_extension(MLIRLinalgPassesBindingsPythonExtension _mlirLinalgPasses
  INSTALL_DIR
    python
  SOURCES
    LinalgPasses.cpp
  LINK_LIBS PRIVATE
    LLVMSupport
    MLIRPythonCAPI
)
add_dependencies(MLIRBindingsPythonExtension MLIRLinalgPassesBindingsPythonExtension)
