add_library(m STATIC
    empty.c
)

install(TARGETS m
    ARCHIVE DESTINATION lib/${NEXT32_TRIPLE}
)
