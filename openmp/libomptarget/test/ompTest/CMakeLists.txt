##===----------------------------------------------------------------------===##
#
# Build actual testing library.
#
##===----------------------------------------------------------------------===##
add_llvm_library(omptest
SHARED

./include/InternalEvent.h
./include/InternalEventCommon.h
./include/OmptAsserter.h
./include/OmptAssertEvent.h
./include/OmptCallbackHandler.h
./include/OmptTester.h
./src/InternalEvent.cpp
./src/InternalEventOperators.cpp
./src/OmptAssertEvent.cpp
./src/OmptCallbackHandler.cpp
./src/OmptTester.cpp

ADDITIONAL_HEADER_DIRS
${LIBOMPTARGET_INCLUDE_DIR}
${LIBOMPTARGET_DEP_LIBFFI_INCLUDE_DIR}

  LINK_LIBS
  PRIVATE
  MemoryManager
  ${LIBOMPTARGET_DEP_LIBFFI_LIBRARIES}
  ${OPENMP_PTHREAD_LIB}

NO_INSTALL_RPATH
)

target_compile_features(omptest PRIVATE cxx_std_17)
