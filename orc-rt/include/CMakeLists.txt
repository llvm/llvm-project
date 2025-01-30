set(ORC_RT_HEADERS
    orc-rt-c/orc-rt.h
)

add_library(orc-rt-headers INTERFACE)
target_include_directories(orc-rt-headers INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
set_property(TARGET orc-rt-headers
    PROPERTY PUBLIC_HEADER ${ORC_RT_HEADERS}
)
install(DIRECTORY ./
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/
    FILES_MATCHING PATTERN "*.h"
)
