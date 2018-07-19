set(LLVM_LINK_COMPONENTS
  Support
  )

set(XRAYSources
 GraphTest.cpp
 )

add_llvm_unittest(XRayTests
    ${XRAYSources}
  )

add_dependencies(XRayTests intrinsics_gen)
