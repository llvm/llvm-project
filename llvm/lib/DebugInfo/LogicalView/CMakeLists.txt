macro(add_lv_impl_folder group)
  list(APPEND LV_IMPL_SOURCES ${ARGN})
  source_group(${group} FILES ${ARGN})
endmacro()

add_lv_impl_folder(Core
  Core/LVOptions.cpp
  )

list(APPEND LIBLV_ADDITIONAL_HEADER_DIRS
  "${LLVM_MAIN_INCLUDE_DIR}/llvm/DebugInfo/LogicalView"
  "${LLVM_MAIN_INCLUDE_DIR}/llvm/DebugInfo/LogicalView/Core"
  )

add_llvm_component_library(LLVMDebugInfoLogicalView
  ${LV_IMPL_SOURCES}

  ADDITIONAL_HEADER_DIRS
  ${LIBLV_ADDITIONAL_HEADER_DIRS}

  LINK_COMPONENTS
  Object
  MC
  Support
  )
