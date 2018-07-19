set(LLVM_LINK_COMPONENTS
  GlobalISel
  CodeGen
  )

add_llvm_unittest(GlobalISelTests
        LegalizerInfoTest.cpp
        )
