file(GLOB_RECURSE SWIG_SOURCES *.swig)

set(FLAGS
  -c++
  -shadow
  -python
  -D__STDC_LIMIT_MACROS
  -D__STDC_CONSTANT_MACROS
  )

set(INCLUDES
  -I${LLDB_SOURCE_DIR}/include
  -I${LLDB_SOURCE_DIR}/tools/intel-features/intel-pt
  )

set(OUTPUT_PYTHON_WRAPPER
  ${CMAKE_CURRENT_BINARY_DIR}/IntelFeaturesPythonWrap.cpp
  )

set(OUTPUT_PYTHON_SCRIPT_DIR
  ${CMAKE_CURRENT_BINARY_DIR}
  )

find_package(SWIG REQUIRED)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/IntelFeaturesPythonWrap.cpp
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/lldbIntelFeatures.py
  DEPENDS ${SWIG_SOURCES}
  COMMAND ${SWIG_EXECUTABLE} ${FLAGS} ${INCLUDES} -o ${OUTPUT_PYTHON_WRAPPER} -outdir ${OUTPUT_PYTHON_SCRIPT_DIR} ${SWIG_SOURCES}
  COMMENT "Generating python wrapper for features library")

set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/IntelFeaturesPythonWrap.cpp PROPERTIES GENERATED 1)
set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/lldbIntelFeatures.py PROPERTIES GENERATED 1)

add_custom_target(intel-features-swig_wrapper ALL
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/IntelFeaturesPythonWrap.cpp
  )
