set(LIBRARY_SOURCES
    atomic.c
    # add other source files here
)

# Compile the library sources to static libraries
foreach(file ${LIBRARY_SOURCES})
    get_filename_component(name ${file} NAME_WE)
    set(out_name ${name}.o)
    set(lib_name next${name})
    add_library(${lib_name} STATIC ${file})

    install(TARGETS ${lib_name}
        ARCHIVE DESTINATION lib/next-ir-lib
    )
endforeach()
