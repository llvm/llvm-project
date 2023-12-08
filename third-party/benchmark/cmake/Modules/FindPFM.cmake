# If successful, the following variables will be defined:
# HAVE_LIBPFM.
# Set BENCHMARK_ENABLE_LIBPFM to 0 to disable, regardless of libpfm presence.
include(CheckIncludeFile)
include(CheckLibraryExists)
include(FeatureSummary)
enable_language(C)

set_package_properties(PFM PROPERTIES
                       URL http://perfmon2.sourceforge.net/
                       DESCRIPTION "a helper library to develop monitoring tools"
                       PURPOSE "Used to program specific performance monitoring events")

check_library_exists(libpfm.a pfm_initialize "" HAVE_LIBPFM_INITIALIZE)
if(HAVE_LIBPFM_INITIALIZE)
  check_include_file(perfmon/perf_event.h HAVE_PERFMON_PERF_EVENT_H)
  check_include_file(perfmon/pfmlib.h HAVE_PERFMON_PFMLIB_H)
  check_include_file(perfmon/pfmlib_perf_event.h HAVE_PERFMON_PFMLIB_PERF_EVENT_H)
  if(HAVE_PERFMON_PERF_EVENT_H AND HAVE_PERFMON_PFMLIB_H AND HAVE_PERFMON_PFMLIB_PERF_EVENT_H)
    message("Using Perf Counters.")
    set(HAVE_LIBPFM 1)
    set(PFM_FOUND 1)
  endif()
else()
  message("Perf Counters support requested, but was unable to find libpfm.")
endif()
