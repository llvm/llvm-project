# Build libxml2 from source when LLVM_BUILD_LIBXML2=TRUE
include(ExternalProject)

set(LIBXML2_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/libxml2/src/libxml2")
set(LIBXML2_BINARY_DIR "${LIBXML2_SOURCE_DIR}-build")

set(libxml2_library_dir "${LIBXML2_BINARY_DIR}/lib/${CMAKE_CFG_INTDIR}")
set(libxml2_library_name libxml2)
if(MSVC)
  set(libxml2_library_name libxml2s)
  set(libxml2_debug_postfix "$<$<CONFIG:Debug>:d>")
endif()
set(LIBXML2_BUILT_LIBRARY "${libxml2_library_dir}/${libxml2_library_name}${CMAKE_STATIC_LIBRARY_SUFFIX}")

if(LLVM_ENABLE_ZLIB)
  # Use the same headers and library selected by LLVM, including for Debug builds.
  set(libxml2_zlib_args
    -DZLIB_INCLUDE_DIR:PATH=${ZLIB_INCLUDE_DIR}
    -DZLIB_LIBRARY:FILEPATH=$<TARGET_LINKER_FILE:ZLIB::ZLIB>
    -DZLIB_LIBRARY_RELEASE:FILEPATH=$<TARGET_LINKER_FILE:ZLIB::ZLIB>
    -DZLIB_LIBRARY_DEBUG:FILEPATH=$<TARGET_LINKER_FILE:ZLIB::ZLIB>)
endif()

ExternalProject_Add(libxml2
  PREFIX libxml2
  GIT_REPOSITORY https://github.com/GNOME/libxml2.git
  GIT_TAG v2.15.1
  GIT_SHALLOW TRUE
  CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
             -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY:PATH=<BINARY_DIR>/lib
             -DBUILD_SHARED_LIBS=OFF
             -DLIBXML2_WITH_PYTHON=OFF
             -DLIBXML2_WITH_PROGRAMS=OFF
             -DLIBXML2_WITH_TESTS=OFF
             # Avoid dynamic deps on ICU / iconv
             -DLIBXML2_WITH_ICU=OFF
             -DLIBXML2_WITH_ICONV=OFF
             -DLIBXML2_WITH_MODULES=OFF
             -DLIBXML2_WITH_ZLIB=${LLVM_ENABLE_ZLIB}
  CMAKE_CACHE_ARGS -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
                   -DCMAKE_TOOLCHAIN_FILE:FILEPATH=${CMAKE_TOOLCHAIN_FILE}
                   -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=${LLVM_ENABLE_PIC}
                   ${libxml2_zlib_args}
  BUILD_BYPRODUCTS "${libxml2_library_dir}/${libxml2_library_name}${libxml2_debug_postfix}${CMAKE_STATIC_LIBRARY_SUFFIX}"
  UPDATE_COMMAND ""
  INSTALL_COMMAND ""
)

# Imported include directories must exist at generation time, before the
# external project's download and configure steps have run.
file(MAKE_DIRECTORY "${LIBXML2_SOURCE_DIR}/include" "${LIBXML2_BINARY_DIR}")
add_library(LibXml2::LibXml2Static STATIC IMPORTED GLOBAL)
set_target_properties(LibXml2::LibXml2Static PROPERTIES
  IMPORTED_LOCATION "${LIBXML2_BUILT_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${LIBXML2_SOURCE_DIR}/include;${LIBXML2_BINARY_DIR}"
)
add_dependencies(LibXml2::LibXml2Static libxml2)

find_package(Threads REQUIRED)
target_link_libraries(LibXml2::LibXml2Static INTERFACE Threads::Threads)
if(LLVM_ENABLE_ZLIB)
  target_link_libraries(LibXml2::LibXml2Static INTERFACE ZLIB::ZLIB)
endif()
if(UNIX)
  target_link_libraries(LibXml2::LibXml2Static INTERFACE m)
endif()
if(WIN32)
  target_link_libraries(LibXml2::LibXml2Static INTERFACE bcrypt)
  target_compile_definitions(LibXml2::LibXml2Static INTERFACE LIBXML_STATIC)
endif()
if(MSVC)
  set_property(TARGET LibXml2::LibXml2Static PROPERTY IMPORTED_LOCATION_DEBUG
    "${libxml2_library_dir}/libxml2sd${CMAKE_STATIC_LIBRARY_SUFFIX}")
endif()
