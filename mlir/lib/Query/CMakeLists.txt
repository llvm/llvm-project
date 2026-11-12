add_mlir_library(MLIRQuery
  Query.cpp
  QueryParser.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Query

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRQueryMatcher
  )

add_subdirectory(Matcher)
