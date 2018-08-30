set(LLVM_LINK_COMPONENTS
  Support
  XRay
  )

add_llvm_unittest(XRayTests
  FDRTraceWriterTest.cpp
  GraphTest.cpp
	ProfileTest.cpp
  )

add_dependencies(XRayTests intrinsics_gen)
