# Multi-threaded test that CAS lock files protecting the shared data are working.

# RUN: rm -rf %t/cas
# RUN: llvm-cas-test -cas %t/cas -check-lock-files
