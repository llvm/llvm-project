if (OPENSCOP_FOUND)
  set(POLLY_OPENSCOP_FILES
      OpenScopImporter.cpp
      OpenScopExporter.cpp)
endif (OPENSCOP_FOUND)

if (SCOPLIB_FOUND)
  set(POLLY_SCOPLIB_FILES
      ScopLib.cpp
      ScopLibExporter.cpp
      ScopLibImporter.cpp)
endif (SCOPLIB_FOUND)

add_polly_library(PollyExchange
  JSONExporter.cpp
  ${POLLY_OPENSCOP_FILES}
  ${POLLY_SCOPLIB_FILES}
)

target_link_libraries(PollyExchange PollyJSON)
