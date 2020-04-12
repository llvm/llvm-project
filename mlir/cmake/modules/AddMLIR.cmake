function(mlir_tablegen ofn)
  tablegen(MLIR ${ARGV})
  set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} ${CMAKE_CURRENT_BINARY_DIR}/${ofn}
      PARENT_SCOPE)
endfunction()

# Declare a dialect in the include directory
function(add_mlir_dialect dialect dialect_namespace)
  set(LLVM_TARGET_DEFINITIONS ${dialect}.td)
  mlir_tablegen(${dialect}.h.inc -gen-op-decls)
  mlir_tablegen(${dialect}.cpp.inc -gen-op-defs)
  mlir_tablegen(${dialect}Dialect.h.inc -gen-dialect-decls -dialect=${dialect_namespace})
  add_public_tablegen_target(MLIR${dialect}IncGen)
  add_dependencies(mlir-headers MLIR${dialect}IncGen)
endfunction()

# Generate Documentation
function(add_mlir_doc doc_filename command output_file output_directory)
  set(LLVM_TARGET_DEFINITIONS ${doc_filename}.td)
  tablegen(MLIR ${output_file}.md ${command} "-I${MLIR_MAIN_SRC_DIR}" "-I${MLIR_INCLUDE_DIR}")
  set(GEN_DOC_FILE ${MLIR_BINARY_DIR}/docs/${output_directory}${output_file}.md)
  add_custom_command(
          OUTPUT ${GEN_DOC_FILE}
          COMMAND ${CMAKE_COMMAND} -E copy
                  ${CMAKE_CURRENT_BINARY_DIR}/${output_file}.md
                  ${GEN_DOC_FILE}
          DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${output_file}.md)
  add_custom_target(${output_file}DocGen DEPENDS ${GEN_DOC_FILE})
  add_dependencies(mlir-doc ${output_file}DocGen)
endfunction()

# Declare a library which can be compiled in libMLIR.so
macro(add_mlir_library name)
  set_property(GLOBAL APPEND PROPERTY MLIR_ALL_LIBS ${name})
  add_llvm_library(${ARGV})
endmacro(add_mlir_library)

# Declare the library associated with a dialect.
function(add_mlir_dialect_library name)
  set_property(GLOBAL APPEND PROPERTY MLIR_DIALECT_LIBS ${name})
  add_mlir_library(${ARGV})
endfunction(add_mlir_dialect_library)

# Declare the library associated with a conversion.
function(add_mlir_conversion_library name)
  set_property(GLOBAL APPEND PROPERTY MLIR_CONVERSION_LIBS ${name})
  add_mlir_library(${ARGV})
endfunction(add_mlir_conversion_library)

# Declare the library associated with a translation.
function(add_mlir_translation_library name)
  set_property(GLOBAL APPEND PROPERTY MLIR_TRANSLATION_LIBS ${name})
  add_mlir_library(${ARGV})
endfunction(add_mlir_translation_library)
