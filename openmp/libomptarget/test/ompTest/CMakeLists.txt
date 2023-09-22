##===----------------------------------------------------------------------===##
#
# Build actual testing library.
#
##===----------------------------------------------------------------------===##
add_llvm_library(omptest
SHARED

./include/InternalEvent.h
./include/OmptTester.h
./include/OmptCallbackHandler.h
./include/OmptAssertEvent.h
./include/OmptAsserter.h
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
