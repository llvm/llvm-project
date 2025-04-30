add_library(next32rt STATIC entrypoint.c)
target_link_libraries(next32rt PRIVATE next32_builtins)

add_library(next32rt_so STATIC shared.c)

install(TARGETS next32rt next32rt_so
    ARCHIVE DESTINATION lib/${NEXT32_TRIPLE}
)
