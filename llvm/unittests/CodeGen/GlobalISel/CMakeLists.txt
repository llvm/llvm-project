set(LLVM_LINK_COMPONENTS
  GlobalISel
  CodeGen
  )

if(LLVM_BUILD_GLOBAL_ISEL)
  add_llvm_unittest(GlobalISelTests
          LegalizerInfoTest.cpp
          )
endif()
