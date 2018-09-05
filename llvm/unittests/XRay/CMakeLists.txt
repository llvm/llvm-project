set(LLVM_LINK_COMPONENTS
  Support
  XRay
  )

add_llvm_unittest(XRayTests
  ProfileTest.cpp
  FDRProducerConsumerTest.cpp
  FDRRecordPrinterTest.cpp
  FDRTraceWriterTest.cpp
  GraphTest.cpp
  )

add_dependencies(XRayTests intrinsics_gen)
