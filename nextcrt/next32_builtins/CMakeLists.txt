set(HEADERS
	global_tables_structure.h
	global_tables.h
	next_memory_structure.h
	next32_pthread.h
	runtime.h
	system_variables.h
)

add_library(next32_builtins INTERFACE)

target_include_directories(next32_builtins INTERFACE
	$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
	$<INSTALL_INTERFACE:include>
)

install(DIRECTORY include/nextcrt DESTINATION include)
