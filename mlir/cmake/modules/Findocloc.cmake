# Findocloc.cmake

# Define a function to search for ocloc
function(find_ocloc)
    message(STATUS "Searching for ocloc")

    if(WIN32)
        if(CMAKE_SIZEOF_VOID_P EQUAL 8)
            set(OCLOC_SUFFIX "64")
        else()
            set(OCLOC_SUFFIX "32")
        endif()

        set(OCLOC_EXE_PATHS "${OCLOC_PACKAGE_DIR}" "${OCLOC_PACKAGE_DIR}/bin")
        set(OCLOC_LIB_PATHS "${OCLOC_PACKAGE_DIR}" "${OCLOC_PACKAGE_DIR}/lib")
        set(OCLOC_INC_PATHS "${OCLOC_PACKAGE_DIR}" "${OCLOC_PACKAGE_DIR}/include")
    else()
        set(OCLOC_SUFFIX "")

        set(OCLOC_EXE_PATHS "${OCLOC_PACKAGE_DIR}/bin")
        set(OCLOC_LIB_PATHS "${OCLOC_PACKAGE_DIR}/lib")
        set(OCLOC_INC_PATHS "${OCLOC_PACKAGE_DIR}/include")

        if(CMAKE_SIZEOF_VOID_P EQUAL 8)
            list(APPEND OCLOC_LIB_PATHS
                "${OCLOC_PACKAGE_DIR}/lib64"
                "${OCLOC_PACKAGE_DIR}/lib/x86_64-linux-gnu")
        endif()
    endif()

    # Search for ocloc executable
    find_program(OCLOC_EXECUTABLE NAMES "ocloc" "ocloc${OCLOC_SUFFIX}"
        PATHS ${OCLOC_EXE_PATHS} NO_DEFAULT_PATH)
    find_program(OCLOC_EXECUTABLE NAMES "ocloc" "ocloc${OCLOC_SUFFIX}"
        PATHS ${OCLOC_EXE_PATHS})

    # Search for ocloc library
    find_library(OCLOC_LIBRARY NAMES "ocloc${OCLOC_SUFFIX}"
        PATHS ${OCLOC_LIB_PATHS} NO_DEFAULT_PATH)
    find_library(OCLOC_LIBRARY NAMES "ocloc${OCLOC_SUFFIX}"
        PATHS ${OCLOC_LIB_PATHS})

    # Search for ocloc_api.h header file
    find_path(OCLOC_INCLUDE_DIR NAMES ocloc_api.h
        PATHS ${OCLOC_INC_PATHS} NO_DEFAULT_PATH)
    find_path(OCLOC_INCLUDE_DIR NAMES ocloc_api.h
        PATHS ${OCLOC_INC_PATHS})

    # Check if all components are found
    if(OCLOC_EXECUTABLE AND OCLOC_LIBRARY AND OCLOC_INCLUDE_DIR)
        set(OCLOC_FOUND TRUE)
    else()
        set(OCLOC_FOUND FALSE)
    endif()

    # Provide the results to the user
    if(OCLOC_FOUND)
        message(STATUS "Found ocloc executable: ${OCLOC_EXECUTABLE}")
        message(STATUS "Found ocloc library: ${OCLOC_LIBRARY}")
        message(STATUS "Found ocloc_api.h: ${OCLOC_INCLUDE_DIR}")
    else()
        message(STATUS "ocloc not found")
    endif()

    # Set the variables for the user
    set(OCLOC_EXECUTABLE ${OCLOC_EXECUTABLE} CACHE FILEPATH "Path to ocloc executable")
    set(OCLOC_LIBRARY ${OCLOC_LIBRARY} CACHE FILEPATH "Path to ocloc library")
    set(OCLOC_INCLUDE_DIR ${OCLOC_INCLUDE_DIR} CACHE PATH "Path to ocloc_api.h header file")
    set(OCLOC_FOUND ${OCLOC_FOUND} CACHE BOOL "ocloc found")

    add_executable(ocloc IMPORTED)
    set_property(TARGET ocloc PROPERTY IMPORTED_LOCATION "${OCLOC_EXECUTABLE}")

    add_library(libocloc SHARED IMPORTED)
    set_target_properties(libocloc PROPERTIES
        IMPORTED_LOCATION "${OCLOC_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${OCLOC_INCLUDE_DIR}")
endfunction()

# Call the function to find ocloc
find_ocloc()

