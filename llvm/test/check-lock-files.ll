# REQUIRES: ondisk_cas

# Multi-threaded test that CAS lock files protecting the shared data are working.

# RUN: rm -rf %t/cas
# RUN: llvm-cas -cas %t/cas -check-lock-files