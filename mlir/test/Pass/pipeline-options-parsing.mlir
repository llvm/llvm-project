// RUN: not mlir-opt %s -pass-pipeline='builtin.module(builtin.module(test-module-pass{))' 2>&1 | FileCheck --check-prefix=CHECK_ERROR_1 %s
// RUN: not mlir-opt %s -pass-pipeline='builtin.module(builtin.module(test-module-pass{test-option=3}))' 2>&1 | FileCheck --check-prefix=CHECK_ERROR_2 %s
// RUN: not mlir-opt %s -pass-pipeline='builtin.module(builtin.module(func.func(test-options-pass{list=3}), test-module-pass{invalid-option=3}))' 2>&1 | FileCheck --check-prefix=CHECK_ERROR_3 %s
// RUN: not mlir-opt %s -pass-pipeline='builtin.module(test-options-pass{list=3 list=notaninteger})' 2>&1 | FileCheck --check-prefix=CHECK_ERROR_4 %s
// RUN: not mlir-opt %s -pass-pipeline='builtin.module(test-options-pass{enum=invalid})' 2>&1 | FileCheck --check-prefix=CHECK_ERROR_5 %s
// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(test-options-pass{list=1,2,3,4 list=5 string=value1 string=value2}))'
// RUN: mlir-opt %s -verify-each=false -pass-pipeline='builtin.module(func.func(test-options-pass{string-list=a list=1,2,3,4 string-list=b,c list=5 string-list=d string=nested_pipeline{arg1=10 arg2=" {} " arg3=true}}))' -dump-pass-pipeline 2>&1 | FileCheck --check-prefix=CHECK_1 %s
// RUN: mlir-opt %s -verify-each=false -test-options-pass-pipeline='list=1 string-list=a,b enum=one' -dump-pass-pipeline 2>&1 | FileCheck --check-prefix=CHECK_2 %s
// RUN: mlir-opt %s -verify-each=false -pass-pipeline='builtin.module(builtin.module(func.func(test-options-pass{list=3}), func.func(test-options-pass{enum=one list=1,2,3,4})))' -dump-pass-pipeline 2>&1 | FileCheck --check-prefix=CHECK_3 %s
// RUN: mlir-opt %s -verify-each=false -pass-pipeline='builtin.module(builtin.module(func.func(test-options-pass{list=3}), func.func(test-options-pass{enum=one list=1,2,3,4 string="foobar"})))' -dump-pass-pipeline 2>&1 | FileCheck --check-prefix=CHECK_4 %s
// RUN: mlir-opt %s -verify-each=false -pass-pipeline='builtin.module(builtin.module(func.func(test-options-pass{list=3}), func.func(test-options-pass{enum=one list=1,2,3,4 string="foo bar baz"})))' -dump-pass-pipeline 2>&1 | FileCheck --check-prefix=CHECK_5 %s
// RUN: mlir-opt %s -verify-each=false -pass-pipeline='builtin.module(builtin.module(func.func(test-options-pass{list=3}), func.func(test-options-pass{enum=one list=1,2,3,4 string={foo bar baz}})))' -dump-pass-pipeline 2>&1 | FileCheck --check-prefix=CHECK_5 %s
// RUN: mlir-opt %s -verify-each=false -pass-pipeline='builtin.module(builtin.module(func.func(test-options-pass{list=3}), func.func(test-options-pass{enum=one list=1,2,3,4 string=foo"bar"baz})))' -dump-pass-pipeline 2>&1 | FileCheck --check-prefix=CHECK_6 %s
// RUN: mlir-opt %s -verify-each=false '-test-options-super-pass-pipeline=super-list={{enum=zero list=1 string=foo},{enum=one list=2 string="bar"},{enum=two list=3 string={baz}}}' -dump-pass-pipeline 2>&1 | FileCheck --check-prefix=CHECK_7 %s
// RUN: mlir-opt %s -verify-each=false -pass-pipeline='builtin.module(func.func(test-options-super-pass{list={{enum=zero list={1} string=foo },{enum=one list={2} string=bar },{enum=two list={3} string=baz }}}))' -dump-pass-pipeline 2>&1 | FileCheck --check-prefix=CHECK_7 %s
// RUN: mlir-opt %s -verify-each=false -test-options-super-set-ab-pipeline='foo=true bar=false' -dump-pass-pipeline 2>&1 | FileCheck --check-prefix=CHECK_11 %s


// This test checks that lists-of-nested-options like 'option1={...},{....}' can be parsed
// just like how 'option=1,2,3' is also allowed:

// RUN: mlir-opt %s -verify-each=false -pass-pipeline='builtin.module(func.func(test-options-super-pass{list={enum=zero list={1} string=foo },{enum=one list={2} string=bar },{enum=two list={3} string=baz }}))' -dump-pass-pipeline 2>&1 | FileCheck --check-prefix=CHECK_7 %s

// This test checks that it is legal to specify an empty list using '{}'.
// RUN: mlir-opt %s -verify-each=false '--test-options-super-pass=list={enum=zero list={1} string=foo},{enum=one list={} string=bar}' -dump-pass-pipeline 2>&1 | FileCheck --check-prefix=CHECK_8 %s

// It is not possible to specify a size-1 list of empty string.
// It is possible to specify a size > 1 list of empty strings.
// RUN: mlir-opt %s -verify-each=false '--pass-pipeline=builtin.module(func.func(test-options-pass{string-list={""}}))' -dump-pass-pipeline 2>&1 | FileCheck --check-prefix=CHECK_9 %s
// RUN: mlir-opt %s -verify-each=false '--pass-pipeline=builtin.module(func.func(test-options-pass{string-list={,}}))' -dump-pass-pipeline 2>&1 | FileCheck --check-prefix=CHECK_10 %s
// RUN: mlir-opt %s -verify-each=false '--pass-pipeline=builtin.module(func.func(test-options-pass{string-list={"",}}))' -dump-pass-pipeline 2>&1 | FileCheck --check-prefix=CHECK_10 %s


// CHECK_ERROR_1: missing closing '}' while processing pass options
// CHECK_ERROR_2: no such option test-option
// CHECK_ERROR_3: no such option invalid-option
// CHECK_ERROR_4: 'notaninteger' value invalid for integer argument
// CHECK_ERROR_5: for the --enum option: Cannot find option named 'invalid'!

// CHECK_1: test-options-pass{enum=zero list={1,2,3,4,5} string=nested_pipeline{arg1=10 arg2=" {} " arg3=true} string-list={a,b,c,d}}
// CHECK_2: test-options-pass{enum=one list={1} string= string-list={a,b}}

// CHECK_3:      builtin.module(
// CHECK_3-NEXT:   builtin.module(
// CHECK_3-NEXT:     func.func(
// CHECK_3-NEXT:       test-options-pass{enum=zero list={3} string= }
// CHECK_3-NEXT:     ),
// CHECK_3-NEXT:     func.func(
// CHECK_3-NEXT:       test-options-pass{enum=one list={1,2,3,4} string= }
// CHECK_3-NEXT:     )
// CHECK_3-NEXT:   )
// CHECK_3-NEXT: )

// CHECK_4:      builtin.module(
// CHECK_4-NEXT:   builtin.module(
// CHECK_4-NEXT:     func.func(
// CHECK_4-NEXT:       test-options-pass{enum=zero list={3} string= }
// CHECK_4-NEXT:     ),
// CHECK_4-NEXT:     func.func(
// CHECK_4-NEXT:       test-options-pass{enum=one list={1,2,3,4} string=foobar }
// CHECK_4-NEXT:     )
// CHECK_4-NEXT:   )
// CHECK_4-NEXT: )

// CHECK_5:      builtin.module(
// CHECK_5-NEXT:   builtin.module(
// CHECK_5-NEXT:     func.func(
// CHECK_5-NEXT:       test-options-pass{enum=zero list={3} string= }
// CHECK_5-NEXT:     ),
// CHECK_5-NEXT:     func.func(
// CHECK_5-NEXT:       test-options-pass{enum=one list={1,2,3,4} string={foo bar baz} }
// CHECK_5-NEXT:     )
// CHECK_5-NEXT:   )
// CHECK_5-NEXT: )

// CHECK_6:      builtin.module(
// CHECK_6-NEXT:   builtin.module(
// CHECK_6-NEXT:     func.func(
// CHECK_6-NEXT:       test-options-pass{enum=zero list={3} string= }
// CHECK_6-NEXT:     ),
// CHECK_6-NEXT:     func.func(
// CHECK_6-NEXT:       test-options-pass{enum=one list={1,2,3,4} string=foo"bar"baz }
// CHECK_6-NEXT:     )
// CHECK_6-NEXT:   )
// CHECK_6-NEXT: )

// CHECK_7{LITERAL}:      builtin.module(
// CHECK_7{LITERAL}-NEXT:   func.func(
// CHECK_7{LITERAL}-NEXT:     test-options-super-pass{list={{enum=zero list={1} string=foo },{enum=one list={2} string=bar },{enum=two list={3} string=baz }}}
// CHECK_7{LITERAL}-NEXT:   )
// CHECK_7{LITERAL}-NEXT: )

// CHECK_8{LITERAL}:      builtin.module(
// CHECK_8{LITERAL}-NEXT:   func.func(
// CHECK_8{LITERAL}-NEXT:     test-options-super-pass{list={{enum=zero list={1} string=foo },{enum=one string=bar }}}
// CHECK_8{LITERAL}-NEXT:   )
// CHECK_8{LITERAL}-NEXT: )

// CHECK_9:      builtin.module(
// CHECK_9-NEXT:   func.func(
// CHECK_9-NEXT:     test-options-pass{enum=zero  string= string-list={}}
// CHECK_9-NEXT:   )
// CHECK_9-NEXT: )

// CHECK_10:      builtin.module(
// CHECK_10-NEXT:   func.func(
// CHECK_10-NEXT:     test-options-pass{enum=zero  string= string-list={,}}
// CHECK_10-NEXT:   )
// CHECK_10-NEXT: )

// CHECK_11:      builtin.module(
// CHECK_11-NEXT:   func.func(
// CHECK_11-NEXT:     test-options-pass-a
// CHECK_11-NEXT:   )
// CHECK_11-NEXT:   func.func(
// CHECK_11-NEXT:     test-options-pass-b
// CHECK_11-NEXT:   )
// CHECK_11-NEXT: )
