set(LLVM_LINK_COMPONENTS Support)

add_llvm_utility(UnicodeNameMappingGenerator
    UnicodeNameMappingGenerator.cpp
    PARTIAL_SOURCES_INTENDED
)

if (NOT LLVM_ENABLE_LIBXML2)
  message(STATUS "Not building UnicodeCharSetsGenerator because libxml2 is not available")
  return()
endif ()

add_llvm_utility(UnicodeCharSetsGenerator
    UnicodeCharSetsGenerator.cpp
    PARTIAL_SOURCES_INTENDED
)
if (LLVM_USE_STATIC_LIBXML2)
  target_link_libraries(UnicodeCharSetsGenerator PRIVATE LibXml2::LibXml2Static)
else()
  target_link_libraries(UnicodeCharSetsGenerator PRIVATE LibXml2::LibXml2)
endif()
