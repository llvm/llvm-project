// RUN: mlir-opt -split-input-file -verify-diagnostics %s

test.op_with_property_predicates <{
  scalar = 1 : i64,
  optional = [2 : i64],
  defaulted = 3 : i64,
  more_constrained = 4 : i64,
  array = [],
  non_empty_unconstrained = [5: i64],
  non_empty_constrained = [6 : i64],
  non_empty_optional = [[7 : i64]],
  unconstrained = 8 : i64}>

// -----

test.op_with_property_predicates <{
  scalar = 1 : i64,
  more_constrained = 4 : i64,
  array = [],
  non_empty_unconstrained = [5: i64],
  non_empty_constrained = [6 : i64],
  unconstrained = 8 : i64}>

// -----

// expected-error @+1 {{'test.op_with_property_predicates' op property 'scalar' failed to satisfy constraint: non-negative int64_t}}
test.op_with_property_predicates <{
  scalar = -1 : i64,
  optional = [2 : i64],
  defaulted = 3 : i64,
  more_constrained = 4 : i64,
  array = [],
  non_empty_unconstrained = [5: i64],
  non_empty_constrained = [6 : i64],
  non_empty_optional = [[7 : i64]],
  unconstrained = 8 : i64}>

// -----

// expected-error @+1 {{'test.op_with_property_predicates' op property 'optional' failed to satisfy constraint: optional non-negative int64_t}}
test.op_with_property_predicates <{
  scalar = 1 : i64,
  optional = [-1 : i64],
  defaulted = 3 : i64,
  more_constrained = 4 : i64,
  array = [],
  non_empty_unconstrained = [5: i64],
  non_empty_constrained = [6 : i64],
  non_empty_optional = [[7 : i64]],
  unconstrained = 8 : i64}>

// -----

// expected-error @+1 {{'test.op_with_property_predicates' op property 'defaulted' failed to satisfy constraint: non-negative int64_t}}
test.op_with_property_predicates <{
  scalar = 1 : i64,
  optional = [2 : i64],
  defaulted = -1 : i64,
  more_constrained = 4 : i64,
  array = [],
  non_empty_unconstrained = [5: i64],
  non_empty_constrained = [6 : i64],
  non_empty_optional = [[7 : i64]],
  unconstrained = 8 : i64}>

// -----

// expected-error @+1 {{'test.op_with_property_predicates' op property 'more_constrained' failed to satisfy constraint: between 0 and 5}}
test.op_with_property_predicates <{
  scalar = 1 : i64,
  optional = [2 : i64],
  defaulted = 3 : i64,
  more_constrained = 100 : i64,
  array = [],
  non_empty_unconstrained = [5: i64],
  non_empty_constrained = [6 : i64],
  non_empty_optional = [[7 : i64]],
  unconstrained = 8 : i64}>

// -----

// expected-error @+1 {{'test.op_with_property_predicates' op property 'array' failed to satisfy constraint: array of non-negative int64_t}}
test.op_with_property_predicates <{
  scalar = 1 : i64,
  optional = [2 : i64],
  defaulted = 3 : i64,
  more_constrained = 4 : i64,
  array = [-1 : i64],
  non_empty_unconstrained = [5: i64],
  non_empty_constrained = [6 : i64],
  non_empty_optional = [[7 : i64]],
  unconstrained = 8 : i64}>

// -----

// expected-error @+1 {{'test.op_with_property_predicates' op property 'non_empty_unconstrained' failed to satisfy constraint: non-empty array of int64_t}}
test.op_with_property_predicates <{
  scalar = 1 : i64,
  optional = [2 : i64],
  defaulted = 3 : i64,
  more_constrained = 4 : i64,
  array = [],
  non_empty_unconstrained = [],
  non_empty_constrained = [6 : i64],
  non_empty_optional = [[7 : i64]],
  unconstrained = 8 : i64}>

// -----

// expected-error @+1 {{'test.op_with_property_predicates' op property 'non_empty_constrained' failed to satisfy constraint: non-empty array of non-negative int64_t}}
test.op_with_property_predicates <{
  scalar = 1 : i64,
  optional = [2 : i64],
  defaulted = 3 : i64,
  more_constrained = 4 : i64,
  array = [],
  non_empty_unconstrained = [5: i64],
  non_empty_constrained = [],
  non_empty_optional = [[7 : i64]],
  unconstrained = 8 : i64}>

// -----

// expected-error @+1 {{'test.op_with_property_predicates' op property 'non_empty_constrained' failed to satisfy constraint: non-empty array of non-negative int64_t}}
test.op_with_property_predicates <{
  scalar = 1 : i64,
  optional = [2 : i64],
  defaulted = 3 : i64,
  more_constrained = 4 : i64,
  array = [],
  non_empty_unconstrained = [5: i64],
  non_empty_constrained = [-1 : i64],
  non_empty_optional = [[7 : i64]],
  unconstrained = 8 : i64}>

// -----

// expected-error @+1 {{'test.op_with_property_predicates' op property 'non_empty_optional' failed to satisfy constraint: optional non-empty array of non-negative int64_t}}
test.op_with_property_predicates <{
  scalar = 1 : i64,
  optional = [2 : i64],
  defaulted = 3 : i64,
  more_constrained = 4 : i64,
  array = [],
  non_empty_unconstrained = [5: i64],
  non_empty_constrained = [6 : i64],
  non_empty_optional = [[]],
  unconstrained = 8 : i64}>
