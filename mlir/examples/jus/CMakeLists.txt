add_custom_target(Jus)
set_target_properties(Jus PROPERTIES FOLDER Examples)

add_dependencies(Jus jusc)
add_llvm_example(jusc
  toyc.cpp 
  parser/AST.cpp
  )

set(LLVM_LINK_COMPONENTS
  Support
  )

include_directories(include/)
target_link_libraries(jusc
  PRIVATE
    MLIRSupport)
