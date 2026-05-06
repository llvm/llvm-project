find_package(Threads REQUIRED)

set(REPLXX_SOURCES
  src/conversion.cxx
  src/ConvertUTF.cpp
  src/escape.cxx
  src/history.cxx
  src/prompt.cxx
  src/replxx_impl.cxx
  src/replxx.cxx
  src/terminal.cxx
  src/util.cxx
  src/wcwidth.cpp
  src/windows.cxx
)

add_library(replxx STATIC ${REPLXX_SOURCES})
add_library(replxx::replxx ALIAS replxx)

target_compile_definitions(replxx PUBLIC REPLXX_STATIC)
target_include_directories(replxx
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
)
set_target_properties(replxx PROPERTIES CXX_STANDARD 11 CXX_STANDARD_REQUIRED ON)
target_link_libraries(replxx PUBLIC Threads::Threads)

if(MSVC)
  target_compile_options(replxx PRIVATE /W0)
else()
  target_compile_options(replxx PRIVATE -w)
endif()
