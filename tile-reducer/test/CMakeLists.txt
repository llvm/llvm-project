configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

add_lit_testsuite(check-tr "Running the TileReducer regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS tr-opt
)
set_target_properties(check-tr PROPERTIES FOLDER "Tests")

add_lit_testsuites(TR ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS tr-opt)
