// REQUIRES: ondisk_cas, shell

// RUN: mkdir -p %t

// This uses a script that triggers parallel `llvm-cas` invocations on an empty directory.
// It's intended to ensure there are no racing issues at creation time that will create failures.
// The last parameter controls how many times it will try.
// To stress-test this extensively change it locally and pass a large number.

// RUN: %S/Inputs/cas-creation-stress-test.sh llvm-cas %t/cas %s 10
