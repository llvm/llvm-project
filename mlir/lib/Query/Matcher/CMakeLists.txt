add_mlir_library(MLIRQueryMatcher
  MatchFinder.cpp
  MatchersInternal.cpp
  Parser.cpp
  RegistryManager.cpp
  VariantValue.cpp
  Diagnostics.cpp
  ErrorBuilder.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Query/Matcher

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRParser
  )
