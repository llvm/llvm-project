# Unlike mlir_float16_utils, mlir_c_runner_utils, etc, we do *not* make
# this a shared library: because doing so causes issues building on Windows.
add_mlir_library(mlir_sparse_tensor_utils
  File.cpp
  NNZ.cpp
  Storage.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  mlir_float16_utils
  )
set_property(TARGET mlir_sparse_tensor_utils PROPERTY CXX_STANDARD 17)

# To make sure we adhere to the style guide:
# <https://llvm.org/docs/CodingStandards.html#provide-a-virtual-method-anchor-for-classes-in-headers>
check_cxx_compiler_flag(-Wweak-vtables
  COMPILER_SUPPORTS_WARNING_WEAK_VTABLES)
if(COMPILER_SUPPORTS_WARNING_WEAK_VTABLES)
  target_compile_options(mlir_sparse_tensor_utils PUBLIC
    "-Wweak-vtables")
endif()
