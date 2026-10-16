set(LLVM_OPTIONAL_SOURCES
  FileUtilities.cpp
  InterfaceSupport.cpp
  StorageUniquer.cpp
  Timing.cpp
  ToolUtilities.cpp
  TypeID.cpp
)

# Keep the complete public library as an umbrella over its components.
add_mlir_library(MLIRSupportCore
  OBJECT
  FileUtilities.cpp
  InterfaceSupport.cpp
  StateStack.cpp
  StorageUniquer.cpp
  Timing.cpp
  TypeID.cpp

  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Support

  LINK_LIBS PUBLIC
  ${LLVM_PTHREAD_LIB}
  )

add_mlir_library(MLIRSupport
  RawOstreamExtras.cpp
  ToolUtilities.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Support

  LINK_LIBS PUBLIC
  MLIRSupportCore
  ${LLVM_PTHREAD_LIB}
  )

# Preserve the complete public archive. The component objects are compiled
# once; libMLIR still collects each object's defining library exactly once.
if(NOT BUILD_SHARED_LIBS)
  target_sources(MLIRSupport PRIVATE $<TARGET_OBJECTS:obj.MLIRSupportCore>)
endif()
