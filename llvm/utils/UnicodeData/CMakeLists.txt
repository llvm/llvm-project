include(CheckCXXSymbolExists)
include(CMakePushCheckState)

set(LLVM_LINK_COMPONENTS Support)

add_llvm_utility(UnicodeNameMappingGenerator
    UnicodeNameMappingGenerator.cpp
    PARTIAL_SOURCES_INTENDED
)

if (NOT LLVM_ENABLE_LIBXML2)
  message(STATUS "Not building UnicodeCharSetsGenerator because libxml2 is not available")
  return()
endif ()

if (LLVM_USE_STATIC_LIBXML2)
  set(libxml2_library LibXml2::LibXml2Static)
else()
  set(libxml2_library LibXml2::LibXml2)
endif()

cmake_push_check_state()
list(APPEND CMAKE_REQUIRED_LIBRARIES ${libxml2_library})
check_cxx_symbol_exists(LIBXML_READER_ENABLED libxml/xmlversion.h HAVE_LIBXML_READER)
cmake_pop_check_state()

if (NOT HAVE_LIBXML_READER)
  message(STATUS "Not building UnicodeCharSetsGenerator because libxml2 reader is not enabled")
  return()
endif ()

add_llvm_utility(UnicodeCharSetsGenerator
    UnicodeCharSetsGenerator.cpp
    PARTIAL_SOURCES_INTENDED
)
target_link_libraries(UnicodeCharSetsGenerator PRIVATE ${libxml2_library})
