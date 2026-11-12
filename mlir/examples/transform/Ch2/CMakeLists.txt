# For a better top-level template to copy, see examples/standalone.

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(include)
add_subdirectory(lib)

add_dependencies(TransformExample transform-opt-ch2)
add_llvm_example(transform-opt-ch2
  transform-opt/transform-opt.cpp)

target_link_libraries(transform-opt-ch2
  PRIVATE
  MLIRIR
  MLIRMlirOptMain
  MLIRSideEffectInterfaces
  MyExtensionCh2
)
