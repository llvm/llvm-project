# - Check if for hash_map.
# CHECK_HASHMAP ()
#

include(CheckCXXSourceCompiles)

macro(CHECK_HASHMAP)
  message(STATUS "Checking for C++ hash_map implementation...")
  check_cxx_source_compiles("
		#include <ext/hash_map>
		int main() {
			__gnu_cxx::hash_map<int, int> t;
		}
"
    HAVE_GNU_EXT_HASH_MAP
    )
  if(HAVE_GNU_EXT_HASH_MAP)
    message(STATUS "C++ hash_map found in 'ext' dir in namespace __gnu_cxx::")
  endif(HAVE_GNU_EXT_HASH_MAP)

  check_cxx_source_compiles("
		#include <ext/hash_map>
		int main() {
			std::hash_map<int, int> t;
		}
"
    HAVE_STD_EXT_HASH_MAP
    )
  if(HAVE_STD_EXT_HASH_MAP)
    message(STATUS "C++ hash_map found in 'ext' dir in namespace std::")
  endif(HAVE_STD_EXT_HASH_MAP)

  check_cxx_source_compiles("
		#include <hash_map>
		int main() {
			hash_map<int, int> t;
		}
"
    HAVE_GLOBAL_HASH_MAP
    )
  if(HAVE_GLOBAL_HASH_MAP)
    message(STATUS "C++ hash_map found in global namespace")
  endif(HAVE_GLOBAL_HASH_MAP)

  if(NOT HAVE_GNU_EXT_HASH_MAP)
    if(NOT HAVE_STD_EXT_HASH_MAP)
      if(NOT HAVE_GLOBAL_HASH_MAP)
	message(STATUS "C++ hash_map not found")
      endif(NOT HAVE_GLOBAL_HASH_MAP)
    endif(NOT HAVE_STD_EXT_HASH_MAP)
  endif(NOT HAVE_GNU_EXT_HASH_MAP)

endmacro(CHECK_HASHMAP)
