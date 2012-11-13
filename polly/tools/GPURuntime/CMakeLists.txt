set(MODULE TRUE)
set(LLVM_NO_RTTI 1)

add_polly_library(GPURuntime
  GPUJIT.c
  )

set_target_properties(GPURuntime
  PROPERTIES
  LINKER_LANGUAGE C
  PREFIX "lib"
  )
