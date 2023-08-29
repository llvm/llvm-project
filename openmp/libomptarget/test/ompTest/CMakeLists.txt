

add_llvm_library(omptest
SHARED

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
  OMPT
  ${LIBOMPTARGET_DEP_LIBFFI_LIBRARIES}
  ${OPENMP_PTHREAD_LIB}

NO_INSTALL_RPATH
)

target_compile_features(omptest PRIVATE cxx_std_17)

add_executable(OMPTesterTest ./src/main.cpp)
target_compile_options(OMPTesterTest PRIVATE "-fopenmp")
target_link_libraries(OMPTesterTest omp omptest)
