#/bin/sh
# Check that isl_test_cpp_failed CANNOT be built.
# Note that the failed build may leave behind a temporary dependence
# tracking object, which should be removed.
make isl_test_cpp_failed
ret=$?
rm -f .deps/isl_test_cpp_failed-isl_test_cpp.Tpo
test $ret -ne 0
