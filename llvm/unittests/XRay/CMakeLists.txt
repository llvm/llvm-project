set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_unittest(XRayTests
  GraphTest.cpp
  )

add_dependencies(XRayTests intrinsics_gen)
