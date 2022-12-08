set(LLVM_OPTIONAL_SOURCES DAdvisor.cpp LowerTypeTests.cpp WholeProgramDevirt.cpp AttributorTest.cpp DynamicallyLoadedInliningAdvisor.cpp)

add_llvm_library(DAdvisor MODULE BUILDTREE_ONLY DAdvisor.cpp)
set_output_directory(DAdvisor
      BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}
      LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}
      )
set_target_properties(DAdvisor PROPERTIES FOLDER "Tests")


set(LLVM_LINK_COMPONENTS
  Analysis
  AsmParser
  Core
  Passes
  IPO
  Support
  TransformUtils
  )

add_llvm_unittest(IPOTests
  LowerTypeTests.cpp
  WholeProgramDevirt.cpp
  AttributorTest.cpp
  DynamicallyLoadedInliningAdvisor.cpp
  )

export_executable_symbols_for_plugins(IPOTests)
add_dependencies(IPOTests DAdvisor)

target_link_libraries(IPOTests PRIVATE LLVMTestingSupport)


set_property(TARGET IPOTests PROPERTY FOLDER "Tests/UnitTests/TransformsTests")
set_property(TARGET DAdvisor PROPERTY FOLDER "Tests/UnitTests/TransformsTests")