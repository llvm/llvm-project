get_property(LLVM_COMMON_DEPENDS GLOBAL PROPERTY LLVM_DRIVER_DEPS)
get_property(LLVM_DRIVER_OBJLIBS GLOBAL PROPERTY LLVM_DRIVER_OBJLIBS)

get_property(LLVM_DRIVER_TOOLS GLOBAL PROPERTY LLVM_DRIVER_TOOLS)

foreach(tool ${LLVM_DRIVER_TOOLS})
  string(REPLACE "-" "_" tool_entry ${tool})
  string(REPLACE "llvm-" "" tool ${tool})
  set(def_decl "${def_decl}LLVM_DRIVER_TOOL(\"${tool}\", ${tool_entry})\n")
endforeach()

get_property(LLVM_EXTRA_DRIVER_ENTRIES GLOBAL PROPERTY LLVM_EXTRA_DRIVER_ENTRIES)

file(WRITE
          "${CMAKE_CURRENT_BINARY_DIR}/LLVMDriverTools.def"
          "${def_decl}${LLVM_EXTRA_DRIVER_ENTRIES}#undef LLVM_DRIVER_TOOL\n")

include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_llvm_tool(llvm-driver
  llvm-driver.cpp
  )

set_target_properties(llvm-driver PROPERTIES OUTPUT_NAME llvm)

target_link_libraries(llvm-driver PUBLIC ${LLVM_DRIVER_OBJLIBS})

if(APPLE)
  # dsymutil uses some CoreFoundation stuff on Darwin...
  target_link_libraries(llvm-driver PRIVATE "-framework CoreFoundation")
endif(APPLE)
