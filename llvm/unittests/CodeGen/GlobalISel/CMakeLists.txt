set(LLVM_LINK_COMPONENTS
  GlobalISel
  )

if(LLVM_BUILD_GLOBAL_ISEL)
  add_llvm_unittest(GlobalISelTests
          MachineLegalizerTest.cpp
          )
endif()
