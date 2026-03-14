// This test should fail. lit used to interpret this as:
//   (false && false) || true
// instead of the intended
//   false && (false || true
//
// RUN: false
// RUN: false || true
//
// XFAIL: *
