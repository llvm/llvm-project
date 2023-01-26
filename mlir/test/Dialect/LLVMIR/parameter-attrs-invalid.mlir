// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// Argument attributes

// expected-error@below {{"llvm.noalias" attribute attached to non-pointer LLVM type}}
llvm.func @invalid_noalias_arg_type(%0 : i32 {llvm.noalias})

// -----

// expected-error@below {{"llvm.noalias" should be a unit attribute}}
llvm.func @invalid_noalias_attr_type(%0 : !llvm.ptr {llvm.noalias = 10 : i32})

// -----

// expected-error@below {{"llvm.readonly" attribute attached to non-pointer LLVM type}}
llvm.func @invalid_readonly_arg_type(%0 : i32 {llvm.readonly})

// -----

// expected-error@below {{"llvm.readonly" should be a unit attribute}}
llvm.func @invalid_readonly_attr_type(%0 : i32 {llvm.readonly = i32})

// -----

// expected-error@below {{"llvm.nest" attribute attached to non-pointer LLVM type}}
llvm.func @invalid_nest_arg_type(%0 : i32 {llvm.nest})

// -----

// expected-error@below {{"llvm.nest" should be a unit attribute}}
llvm.func @invalid_nest_attr_type(%0 : i32 {llvm.nest = "foo"})

// -----

// expected-error@below {{"llvm.align" attribute attached to non-pointer LLVM type}}
llvm.func @invalid_align_arg_type(%0 : i32 {llvm.align = 10 : i32})

// -----

// expected-error@below {{"llvm.align" should be an integer attribute}}
llvm.func @invalid_align_attr_type(%0 : i32 {llvm.align = "foo"})

// -----

// expected-error@below {{"llvm.sret" attribute attached to non-pointer LLVM type}}
llvm.func @invalid_sret_arg_type(%0 : i32 {llvm.sret = !llvm.struct<(i32)>})

// -----

// expected-error@below {{"llvm.sret" attribute attached to LLVM pointer argument of different type}}
llvm.func @invalid_sret_attr_type(%0 : !llvm.ptr<f32> {llvm.sret = !llvm.struct<(i32)>})

// -----

// expected-error@below {{"llvm.byval" attribute attached to non-pointer LLVM type}}
llvm.func @invalid_byval_arg_type(%0 : i32 {llvm.byval = !llvm.struct<(i32)>})

// -----

// expected-error@below {{"llvm.byval" attribute attached to LLVM pointer argument of different type}}
llvm.func @invalid_byval_attr_type(%0 : !llvm.ptr<!llvm.struct<(f32)>> {llvm.byval = !llvm.struct<(i32)>})

// -----

// expected-error@below {{"llvm.byref" attribute attached to non-pointer LLVM type}}
llvm.func @invalid_byref_arg_type(%0 : i32 {llvm.byref = !llvm.struct<(i32)>})

// -----

// expected-error@below {{"llvm.byref" attribute attached to LLVM pointer argument of different type}}
llvm.func @invalid_byref_attr_type(%0 : !llvm.ptr<!llvm.struct<(f32)>> {llvm.byref = !llvm.struct<(i32)>})

// -----

// expected-error@below {{"llvm.inalloca" attribute attached to non-pointer LLVM type}}
llvm.func @invalid_inalloca_arg_type(%0 : i32 {llvm.inalloca = !llvm.struct<(i32)>})

// -----

// expected-error@below {{"llvm.inalloca" attribute attached to LLVM pointer argument of different type}}
llvm.func @invalid_inalloca_attr_type(%0 : !llvm.ptr<!llvm.struct<(f32)>> {llvm.inalloca = !llvm.struct<(i32)>})

// -----

// expected-error@below {{"llvm.signext" attribute attached to non-integer LLVM type}}
llvm.func @invalid_signext_arg_type(%0 : f32 {llvm.signext})

// -----

// expected-error@below {{"llvm.signext" should be a unit attribute}}
llvm.func @invalid_signext_attr_type(%0 : i32 {llvm.signext = !llvm.struct<(i32)>})

// -----

// expected-error@below {{"llvm.zeroext" attribute attached to non-integer LLVM type}}
llvm.func @invalid_zeroext_arg_type(%0 : f32 {llvm.zeroext})

// -----

// expected-error@below {{"llvm.zeroext" should be a unit attribute}}
llvm.func @invalid_zeroext_attr_type(%0 : i32 {llvm.zeroext = !llvm.struct<(i32)>})

// -----

// expected-error@below {{"llvm.noundef" should be a unit attribute}}
llvm.func @invalid_noundef_attr_type(%0 : i32 {llvm.noundef = !llvm.ptr})

// -----

// Result attributes

// expected-error@below {{cannot attach result attributes to functions with a void return}}
llvm.func @void_def() -> (!llvm.void {llvm.noundef})

// -----

// expected-error @below{{"llvm.align" should be an integer attribute}}
llvm.func @alignattr_ret() -> (!llvm.ptr {llvm.align = 1.0 : f32})

// -----

// expected-error @below{{"llvm.align" attribute attached to non-pointer LLVM type}}
llvm.func @alignattr_ret() -> (i32 {llvm.align = 4})

// -----

// expected-error @below{{"llvm.noalias" should be a unit attribute}}
llvm.func @noaliasattr_ret() -> (!llvm.ptr {llvm.noalias = 1})

// -----

// expected-error @below{{"llvm.noalias" attribute attached to non-pointer LLVM type}}
llvm.func @noaliasattr_ret() -> (i32 {llvm.noalias})

// -----

// expected-error @below{{"llvm.noundef" should be a unit attribute}}
llvm.func @noundefattr_ret() -> (!llvm.ptr {llvm.noundef = 1})

// -----

// expected-error @below{{"llvm.signext" should be a unit attribute}}
llvm.func @signextattr_ret() -> (i32 {llvm.signext = 1})

// -----

// expected-error @below{{"llvm.signext" attribute attached to non-integer LLVM type}}
llvm.func @signextattr_ret() -> (f32 {llvm.signext})

// -----

// expected-error @below{{"llvm.zeroext" should be a unit attribute}}
llvm.func @zeroextattr_ret() -> (i32 {llvm.zeroext = 1})

// -----

// expected-error @below{{"llvm.zeroext" attribute attached to non-integer LLVM type}}
llvm.func @zeroextattr_ret() -> (f32 {llvm.zeroext})

// -----

// expected-error @below{{"llvm.readonly" is not a valid result attribute}}
llvm.func @readonly_ret() -> (f32 {llvm.readonly})

// -----

// expected-error @below{{"llvm.nest" is not a valid result attribute}}
llvm.func @nest_ret() -> (f32 {llvm.nest})

// -----

// expected-error @below{{"llvm.sret" is not a valid result attribute}}
llvm.func @sret_ret() -> (!llvm.ptr {llvm.sret = i64})

// -----

// expected-error @below{{"llvm.byval" is not a valid result attribute}}
llvm.func @byval_ret() -> (!llvm.ptr {llvm.byval = i64})

// -----

// expected-error @below{{"llvm.byref" is not a valid result attribute}}
llvm.func @byref_ret() -> (!llvm.ptr {llvm.byref = i64})

// -----

// expected-error @below{{"llvm.inalloca" is not a valid result attribute}}
llvm.func @inalloca_ret() -> (!llvm.ptr {llvm.inalloca = i64})
