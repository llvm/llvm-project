set(files
    orc-rt-c/orc-rt.h
)

add_library(orc-rt-headers INTERFACE)
target_include_directories(orc-rt-headers INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_sources(orc-rt-headers
    INTERFACE FILE_SET HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
    FILES ${files}
)
install(TARGETS orc-rt-headers
    FILE_SET HEADERS DESTINATION include)
