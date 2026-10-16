if(MLIR_ENABLE_PDL_IN_PATTERNMATCH)
  set(pdl_src
    PDL/PDLPatternMatch.cpp
  )
endif()

# Keep the complete public library as an umbrella over its components.
add_mlir_library(MLIRIRCore
  OBJECT
  AffineExpr.cpp
  AffineMap.cpp
  AsmPrinter.cpp
  Attributes.cpp
  AttrTypeSubElements.cpp
  Block.cpp
  Builders.cpp
  BuiltinAttributeInterfaces.cpp
  BuiltinAttributes.cpp
  BuiltinDialect.cpp
  BuiltinDialectBytecode.cpp
  BuiltinTypes.cpp
  BuiltinTypeInterfaces.cpp
  Diagnostics.cpp
  Dialect.cpp
  DialectResourceBlobManager.cpp
  Dominance.cpp
  ExtensibleDialect.cpp
  IntegerSet.cpp
  Location.cpp
  MLIRContext.cpp
  ODSSupport.cpp
  Operation.cpp
  OperationSupport.cpp
  PatternMatch.cpp
  Region.cpp
  RegionKindInterface.cpp
  Remarks.cpp
  SymbolTable.cpp
  TensorEncoding.cpp
  Types.cpp
  TypeRange.cpp
  TypeUtilities.cpp
  Value.cpp
  ValueRange.cpp
  Verifier.cpp
  Visitors.cpp
  ${pdl_src}

  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/IR

  PRECOMPILE_HEADERS
  [["mlir/IR/pch.h"]]

  DEPENDS
  MLIRBuiltinAttributesIncGen
  MLIRBuiltinAttributeInterfacesIncGen
  MLIRBuiltinDialectBytecodeIncGen
  MLIRBuiltinDialectIncGen
  MLIRBuiltinLocationAttributesIncGen
  MLIRBuiltinOpsIncGen
  MLIRBuiltinTypesIncGen
  MLIRBuiltinTypeConstraintsIncGen
  MLIRBuiltinTypeInterfacesIncGen
  MLIRDialectFoldInterfaceIncGen
  MLIROpAsmInterfaceIncGen
  MLIRQuantStorageTypeInterfaceIncGen
  MLIRRegionKindInterfaceIncGen
  MLIRSymbolInterfacesIncGen
  MLIRTensorEncodingIncGen
  MLIROpAsmDialectInterfaceIncGen
  MLIRBytecodeDialectInterfaceIncGen
  MLIRBytecodeOpInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRSupportCore
  )

# PatternApplicator uses this listener when NDEBUG is not defined.
add_mlir_library(MLIRIRPatternLogging
  OBJECT
  PatternLoggingListener.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  MLIRIRCore
  )

add_mlir_library(MLIRIR
  QuantStorageTypeInterface.cpp
  Unit.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/IR

  DEPENDS
  MLIRBuiltinAttributesIncGen
  MLIRBuiltinAttributeInterfacesIncGen
  MLIRBuiltinDialectBytecodeIncGen
  MLIRBuiltinDialectIncGen
  MLIRBuiltinLocationAttributesIncGen
  MLIRBuiltinOpsIncGen
  MLIRBuiltinTypesIncGen
  MLIRBuiltinTypeConstraintsIncGen
  MLIRBuiltinTypeInterfacesIncGen
  MLIRDialectFoldInterfaceIncGen
  MLIROpAsmInterfaceIncGen
  MLIRQuantStorageTypeInterfaceIncGen
  MLIRRegionKindInterfaceIncGen
  MLIRSymbolInterfacesIncGen
  MLIRTensorEncodingIncGen
  MLIROpAsmDialectInterfaceIncGen
  MLIRBytecodeDialectInterfaceIncGen
  MLIRBytecodeOpInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRIRCore
  MLIRIRPatternLogging
  MLIRSupport
  )

# Preserve the complete public archive. The component objects are compiled
# once; libMLIR still collects each object's defining library exactly once.
if(NOT BUILD_SHARED_LIBS)
  target_sources(MLIRIR PRIVATE
    $<TARGET_OBJECTS:obj.MLIRIRCore>
    $<TARGET_OBJECTS:obj.MLIRIRPatternLogging>)
endif()
