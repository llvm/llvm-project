set(LLVM_OPTIONAL_SOURCES
  PythonTestCAPI.cpp
  PythonTestDialect.cpp
  PythonTestModule.cpp
)

add_mlir_library(MLIRPythonTestDialect
  PythonTestDialect.cpp

  EXCLUDE_FROM_LIBMLIR

  DEPENDS
  MLIRPythonTestIncGen

  LINK_LIBS PUBLIC
  MLIRInferTypeOpInterface
  MLIRIR
  MLIRSupport
)

add_mlir_public_c_api_library(MLIRCAPIPythonTestDialect
  PythonTestCAPI.cpp

  DEPENDS
  MLIRPythonTestIncGen

  LINK_LIBS PUBLIC
  MLIRCAPIInterfaces
  MLIRCAPIIR
  MLIRCAPIRegistration
  MLIRPythonTestDialect
)

