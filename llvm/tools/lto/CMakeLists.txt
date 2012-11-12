set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  ipo scalaropts linker bitreader bitwriter mcdisassembler vectorize)

add_definitions( -DLLVM_VERSION_INFO=\"${PACKAGE_VERSION}\" )

set(SOURCES
  LTOCodeGenerator.cpp
  lto.cpp
  LTOModule.cpp
  )

if( NOT WIN32 AND LLVM_ENABLE_PIC )
  set(bsl ${BUILD_SHARED_LIBS})
  set(BUILD_SHARED_LIBS ON)
  add_llvm_library(LTO ${SOURCES})
  set_property(TARGET LTO PROPERTY OUTPUT_NAME "LTO")
  set(BUILD_SHARED_LIBS ${bsl})
  set(LTO_STATIC_TARGET_NAME LTO_static)
else()
  set(LTO_STATIC_TARGET_NAME LTO)
endif()

if( NOT BUILD_SHARED_LIBS )
  add_llvm_library(${LTO_STATIC_TARGET_NAME} ${SOURCES})
  set_property(TARGET ${LTO_STATIC_TARGET_NAME} PROPERTY OUTPUT_NAME "LTO")
endif()
