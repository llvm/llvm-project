set(LLVM_LINK_COMPONENTS
  Support
  )

add_kaleidoscope_chapter(Kaleidoscope-Ch2
  toy.cpp
  )

if(LLVM_COMPILER_IS_GCC_COMPATIBLE)
  target_compile_options(Kaleidoscope-Ch2 PRIVATE
    -Wno-unused-private-field
    )
endif()
