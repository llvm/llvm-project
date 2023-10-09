add_custom_target(Lisp)
set_target_properties(Lisp PROPERTIES FOLDER Examples)


add_dependencies(Lisp lispc)
add_llvm_example(lispc lispc.cpp parser/AST.cpp)

set(LLVM_LINK_COMPONENTS
  Support
  )


include_directories(include/)
target_link_libraries(lispc
  PRIVATE
    MLIRSupport)
