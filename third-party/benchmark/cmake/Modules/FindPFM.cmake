# If successful, the following variables will be defined:
# PFM_FOUND.
# PFM_LIBRARIES
# PFM_INCLUDE_DIRS
# the following target will be defined:
# PFM::libpfm

include(FeatureSummary)
include(FindPackageHandleStandardArgs)

set_package_properties(PFM PROPERTIES
                       URL http://perfmon2.sourceforge.net/
                       DESCRIPTION "A helper library to develop monitoring tools"
                       PURPOSE "Used to program specific performance monitoring events")

find_library(PFM_LIBRARY NAMES pfm)
find_path(PFM_INCLUDE_DIR NAMES perfmon/pfmlib.h)

find_package_handle_standard_args(PFM REQUIRED_VARS PFM_LIBRARY PFM_INCLUDE_DIR)

if (PFM_FOUND AND NOT TARGET PFM::libpfm)
    add_library(PFM::libpfm UNKNOWN IMPORTED)
    set_target_properties(PFM::libpfm PROPERTIES
        IMPORTED_LOCATION "${PFM_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${PFM_INCLUDE_DIR}")
endif()

mark_as_advanced(PFM_LIBRARY PFM_INCLUDE_DIR)
