# - Check if for hash_set.
# CHECK_HASHSET ()
#

include(CheckCXXSourceCompiles)

macro(CHECK_HASHSET)
  message(STATUS "Checking for C++ hash_set implementation...")
  check_cxx_source_compiles("
		#include <ext/hash_set>
		int main() {
			__gnu_cxx::hash_set<int> t;
		}
"
    HAVE_GNU_EXT_HASH_SET
    )
  if(HAVE_GNU_EXT_HASH_SET)
    message(STATUS "C++ hash_set found in 'ext' dir in namespace __gnu_cxx::")
  endif(HAVE_GNU_EXT_HASH_SET)

  check_cxx_source_compiles("
		#include <ext/hash_set>
		int main() {
			std::hash_set<int> t;
		}
"
    HAVE_STD_EXT_HASH_SET
    )
  if(HAVE_STD_EXT_HASH_SET)
    message(STATUS "C++ hash_set found in 'ext' dir in namespace std::")
  endif(HAVE_STD_EXT_HASH_SET)

  check_cxx_source_compiles("
		#include <hash_set>
		int main() {
			hash_set<int> t;
		}
"
    HAVE_GLOBAL_HASH_SET
    )
  if(HAVE_GLOBAL_HASH_SET)
    message(STATUS "C++ hash_set found in global namespace")
  endif(HAVE_GLOBAL_HASH_SET)

  if(NOT HAVE_GNU_EXT_HASH_SET)
    if(NOT HAVE_STD_EXT_HASH_SET)
      if(NOT HAVE_GLOBAL_HASH_SET)
	message(STATUS "C++ hash_set not found")
      endif(NOT HAVE_GLOBAL_HASH_SET)
    endif(NOT HAVE_STD_EXT_HASH_SET)
  endif(NOT HAVE_GNU_EXT_HASH_SET)
endmacro(CHECK_HASHSET)
