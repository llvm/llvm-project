## Check escaping of [[, which can occur in global initializers.
# RUN: cp -f %S/Inputs/placeholder_escape.ll %t.ll && %update_test_checks %t.ll --check-globals
# RUN: diff -u %t.ll %S/Inputs/placeholder_escape.ll.expected
