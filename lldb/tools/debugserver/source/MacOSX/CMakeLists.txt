list(APPEND SOURCES arm/DNBArchImpl.cpp arm64/DNBArchImplARM64.cpp)
include_directories(${CURRENT_SOURCE_DIR}/arm ${CURRENT_SOURCE_DIR}/arm64)

list(APPEND SOURCES i386/DNBArchImplI386.cpp x86_64/DNBArchImplX86_64.cpp)
include_directories(${CURRENT_SOURCE_DIR}/i386 ${CURRENT_SOURCE_DIR}/x86_64)

include_directories(..)

include_directories(${LLDB_SOURCE_DIR}/tools/debugserver/source)
add_library(lldbDebugserverArchSupport
  ${SOURCES}
  )

set_target_properties(lldbDebugserverArchSupport PROPERTIES FOLDER "lldb libraries/debugserver")
