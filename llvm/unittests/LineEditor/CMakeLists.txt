set(LLVM_LINK_COMPONENTS
  LineEditor
  Support
  )

add_llvm_unittest(LineEditorTests
  LineEditor.cpp
  )
