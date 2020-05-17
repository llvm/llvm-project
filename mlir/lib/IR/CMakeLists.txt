add_mlir_library(MLIRIR
  AffineExpr.cpp
  AffineMap.cpp
  AsmPrinter.cpp
  Attributes.cpp
  Block.cpp
  Builders.cpp
  Diagnostics.cpp
  Dialect.cpp
  Dominance.cpp
  Function.cpp
  FunctionImplementation.cpp
  IntegerSet.cpp
  Location.cpp
  MLIRContext.cpp
  Module.cpp
  Operation.cpp
  OperationSupport.cpp
  PatternMatch.cpp
  Region.cpp
  StandardTypes.cpp
  SymbolTable.cpp
  Types.cpp
  TypeUtilities.cpp
  Value.cpp
  Verifier.cpp
  Visitors.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/IR

  DEPENDS
  MLIRCallInterfacesIncGen
  MLIROpAsmInterfacesIncGen
  MLIRSymbolInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRSupport
  )
