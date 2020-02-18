set(LLVM_OPTIONAL_SOURCES
  cblas.cpp
  cblas_interface.cpp
  )

add_llvm_library(cblas SHARED cblas.cpp)
target_compile_definitions(cblas PRIVATE cblas_EXPORTS)

add_llvm_library(cblas_interface SHARED cblas_interface.cpp
  LINK_LIBS
  cblas
  )
target_compile_definitions(cblas_interface PRIVATE cblas_interface_EXPORTS)

