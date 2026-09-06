add_llvm_unittest(LTOTests lto-c-api.cpp)
target_link_libraries(LTOTests PRIVATE LTO)

