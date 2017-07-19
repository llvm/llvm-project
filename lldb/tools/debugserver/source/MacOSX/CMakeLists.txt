if("${CMAKE_OSX_ARCHITECTURES}" MATCHES ".*arm.*")
  list(APPEND SOURCES arm/DNBArchImpl.cpp arm64/DNBArchImplARM64.cpp)
  include_directories(${CURRENT_SOURCE_DIR}/arm ${CURRENT_SOURCE_DIR}/arm64)
endif()

if(NOT CMAKE_OSX_ARCHITECTURES OR "${CMAKE_OSX_ARCHITECTURES}" MATCHES ".*86.*")
  list(APPEND SOURCES i386/DNBArchImplI386.cpp x86_64/DNBArchImplX86_64.cpp)
  include_directories(${CURRENT_SOURCE_DIR}/i386 ${CURRENT_SOURCE_DIR}/x86_64)
endif()

if("${CMAKE_OSX_ARCHITECTURES}" MATCHES ".*ppc.*")
  list(APPEND SOURCES ppc/DNBArchImpl.cpp)
  include_directories(${CURRENT_SOURCE_DIR}/ppc)
endif()

add_subdirectory(DarwinLog)

include_directories(..)

include_directories(${LLDB_SOURCE_DIR}/tools/debugserver/source)
add_library(lldbDebugserverArchSupport
  ${SOURCES}
  )
