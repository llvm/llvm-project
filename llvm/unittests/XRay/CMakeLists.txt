set(LLVM_LINK_COMPONENTS
  Support
  XRay
  TestingSupport
  )

add_llvm_unittest(XRayTests
  ProfileTest.cpp
  FDRBlockIndexerTest.cpp
  FDRBlockVerifierTest.cpp
  FDRProducerConsumerTest.cpp
  FDRRecordPrinterTest.cpp
  FDRTraceWriterTest.cpp
  GraphTest.cpp
  )

add_dependencies(XRayTests intrinsics_gen)
